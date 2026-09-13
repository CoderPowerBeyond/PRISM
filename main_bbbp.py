import os

import copy

import time

import argparse

from utils.config_utils import parse_args_with_gat_config

import random

import numpy as np

from collections import defaultdict

from rdkit import Chem

from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.Chem import AllChem

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedShuffleSplit

import torch

import torch.nn as nn

from torch_geometric.datasets import MoleculeNet

from torch_geometric.loader import DataLoader

from model.sdn import EnhancedOG_PGAT

try:

    from model.mace import build_mace_for_geomo

except ImportError:

    build_mace_for_geomo = None

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

    """Split dataset by molecular scaffold, fallback to stratified if needed."""

    n = len(dataset)

    labels = []

    for d in dataset:

        y = d.y.view(-1).numpy()

        y = y[y != -1]

        labels.append(float(np.nanmean(y > 0.5)) if len(y) else 0.0)

    labels = np.array(labels)

    scaffolds = defaultdict(list)

    for i, data in enumerate(dataset):

        smi = getattr(data, "smiles", None)

        mol = Chem.MolFromSmiles(smi) if smi else None

        if mol is None:

            scaffolds["invalid"].append(i)

            continue

        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

        scaffolds[scaf].append(i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)

    n_train, n_val = int(frac_train * n), int(frac_val * n)

    train, val, test = [], [], []

    for g in groups:

        if len(train) + len(g) <= n_train:

            train += g

        elif len(val) + len(g) <= n_val:

            val += g

        else:

            test += g

    def single_class(idxs):

        labs = labels[idxs]

        labs = labs[~np.isnan(labs)]

        return len(np.unique(labs)) < 2

    if not train or single_class(val) or single_class(test):

        print("Switching to stratified fallback split.")

        X = np.zeros((len(labels), 1))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac_val + frac_test, random_state=seed)

        train_idx, temp_idx = next(sss.split(X, labels))

        tval = frac_val / (frac_val + frac_test)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tval, random_state=seed)

        val_idx, test_idx = next(sss2.split(X[temp_idx], labels[temp_idx]))

        val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

        train, val, test = train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    return train, val, test

def smiles_to_3d_pos(smiles: str, num_heavy_atoms: int = None):

    """3D coords for heavy atoms only (matches MoleculeNet node count)."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    n_heavy = mol.GetNumAtoms()

    mol_h = Chem.AddHs(mol)

    try:

        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

        if res != 0 or mol_h.GetNumConformers() == 0:

            return None

        AllChem.UFFOptimizeMolecule(mol_h, maxIters=100)

        conf = mol_h.GetConformer()

        coords = [list(conf.GetAtomPosition(i)) for i in range(n_heavy)]

        pos = torch.tensor(coords, dtype=torch.float)

        if num_heavy_atoms is not None and pos.shape[0] != num_heavy_atoms:

            return None

        return pos

    except Exception:

        return None

def smiles_to_atomic_numbers(smiles: str, num_heavy_atoms: int = None):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    n_heavy = mol.GetNumAtoms()

    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_heavy)], dtype=torch.long)

    if num_heavy_atoms is not None and z.shape[0] != num_heavy_atoms:

        return None

    return z

def attach_positions(dataset, attach_z_for_mace: bool = True):

    """Cache pos/z per index (PyG dataset[i] often drops pos). MoleculeNet = heavy atoms only."""

    dataset._geo_pos = {}

    dataset._geo_z = {}

    missing, total = 0, len(dataset)

    for i in range(total):

        data = dataset[i]

        smi = getattr(data, "smiles", None)

        nh = data.x.shape[0]

        pos = smiles_to_3d_pos(smi, num_heavy_atoms=nh) if smi else None

        if pos is not None:

            dataset._geo_pos[i] = pos

        else:

            missing += 1

        if attach_z_for_mace and smi is not None:

            z = smiles_to_atomic_numbers(smi, num_heavy_atoms=nh)

            if z is not None:

                dataset._geo_z[i] = z

    print(

        f"Added 3D coordinates to {total - missing}/{total} molecules "

        f"({missing} without valid conformer)."

    )

    return dataset

def dataset_sample_with_geom(dataset, idx):

    d = dataset[idx].clone()

    g = getattr(dataset, "_geo_pos", None)

    if g is not None and idx in g:

        d.pos = g[idx]

    gz = getattr(dataset, "_geo_z", None)

    if gz is not None and idx in gz:

        d.z = gz[idx]

    return d

def reasoning_loss(model):

    """Safely extract curvature regularization term (LB + DE)."""

    try:

        return model.get_curvature_reg()

    except Exception:

        return torch.tensor(0.0, device=next(model.parameters()).device)

def find_convergence_epoch(values, window=10, rel_threshold=0.05):

    """Find the first epoch where values stabilize for `window` consecutive epochs."""

    for start in range(len(values) - window):

        ref = values[start]

        if abs(ref) < 1e-10:

            continue

        segment = values[start:start + window]

        max_rel_change = max(abs(v - ref) / abs(ref) for v in segment)

        if max_rel_change < rel_threshold:

            return start + 1

    return None

def train_epoch(model, loader, optimizer, device, lam, pos_weight=None, noise_level=0.0):

    model.train()

    total_loss, count = 0.0, 0

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn_like(batch.x)

        optimizer.zero_grad()

        out, kappa = model(batch)

        if out.dim() > 1 and batch.y.dim() > 1:

            if out.shape[1] > batch.y.shape[1]:

                out = out[:, :batch.y.shape[1]]

        out = out.float().clamp(-20, 20)

        y = batch.y.float().view(out.shape[0], -1)

        mask = y != -1

        if mask.sum() == 0:

            continue

        loss_mat = criterion(out, y.nan_to_num(0.0))

        if pos_weight is not None:

            w = pos_weight.view(1, -1).expand_as(y)

            loss_mat = loss_mat * torch.where(y > 0.5, w, 1.0)

        loss_main = loss_mat[mask].mean()

        reg_loss = reasoning_loss(model)

        if not torch.isfinite(reg_loss):

            reg_loss = torch.tensor(0.0, device=device)

        loss = loss_main + lam * reg_loss

        if not torch.isfinite(loss):

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total_loss += loss.item()

        count += 1

    return total_loss / max(count, 1)

@torch.no_grad()

def eval_epoch(model, loader, device, num_tasks, noise_level=0.0):

    model.eval()

    y_true, y_pred = [], []

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn_like(batch.x)

        out, _ = model(batch)

        if out.dim() > 1 and batch.y.dim() > 1:

            if out.shape[1] > batch.y.shape[1]:

                out = out[:, :batch.y.shape[1]]

        out = out.float().clamp(-20, 20)

        y = batch.y.float().view(out.shape[0], -1)

        mask = y != -1

        if mask.sum() == 0:

            continue

        y_true.append(y[mask].cpu())

        y_pred.append(torch.sigmoid(out[mask]).cpu())

    if not y_true:

        return 0.0

    y_true, y_pred = torch.cat(y_true, 0).numpy(), torch.cat(y_pred, 0).numpy()

    def safe_auc(y, p):

        if len(np.unique(y)) < 2:

            return np.nan

        return roc_auc_score(y, p)

    aucs = []

    try:

        if y_true.ndim == 1 or y_true.shape[1] == 1:

            a1, a2 = safe_auc(y_true, y_pred), safe_auc(y_true, 1 - y_pred)

            return float(np.nanmax([a1, a2]))

        for i in range(y_true.shape[1]):

            yi, pi = y_true[:, i], y_pred[:, i]

            if len(np.unique(yi)) < 2:

                continue

            a1, a2 = safe_auc(yi, pi), safe_auc(yi, 1 - pi)

            aucs.append(np.nanmax([a1, a2]))

        return float(np.nanmean(aucs)) if aucs else 0.0

    except Exception:

        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bace", choices=["bbbp", "bace", "tox21", "clintox", "hiv", "sider", "muv"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lambda_reason", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2046)
    parser.add_argument("--model", choices=("geomo", "mace"), default="geomo",
                        help="geomo=EnhancedOG_PGAT, mace=MACE baseline (requires pip install mace-torch)")
    # 新增保存目录参数
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=0,
                        help="If > 0, also save a periodic checkpoint every N epochs")
    parser.add_argument("--ckpt", default="", help="Checkpoint path. Used by --eval_only, or to resume weights.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and evaluate checkpoint on val/test")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Repeat training with seeds seed, seed+1, ... and report mean±std")
    args = parse_args_with_gat_config(parser)

    os.makedirs(args.save_dir, exist_ok=True)

    ckpt_path = args.ckpt or os.path.join(args.save_dir, f"{args.dataset}_best_model.pth")
    ckpt = None
    if args.eval_only or args.ckpt:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        if isinstance(saved_args, argparse.Namespace):
            saved_args = vars(saved_args)
        for key in ("dataset", "hidden", "layers", "dropout", "model", "seed"):
            if key in saved_args:
                setattr(args, key, saved_args[key])
        print(f"Loaded checkpoint metadata from {ckpt_path} "
              f"(epoch={ckpt.get('epoch')}, best_val_auc={ckpt.get('best_val_auc')})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MoleculeNet(root=f"data/{args.dataset}", name=args.dataset)

    print(f"Loaded {args.dataset.upper()} with {len(dataset)} molecules")

    dataset = attach_positions(dataset)

    num_feats = dataset.num_node_features

    num_edge_feats = dataset.num_edge_features if hasattr(dataset, 'num_edge_features') else 3

    num_tasks = dataset.num_classes or 1

    num_tasks = max(num_tasks, 1)

    def make_split(seed):
        tr_i, va_i, te_i = scaffold_split(dataset, seed=seed)
        has_geo = set(getattr(dataset, "_geo_pos", {}).keys())
        if len(has_geo) < len(dataset):
            tr_i = [i for i in tr_i if i in has_geo]
            va_i = [i for i in va_i if i in has_geo]
            te_i = [i for i in te_i if i in has_geo]
            if not tr_i or not va_i or not te_i:
                raise RuntimeError("A split is empty after filtering missing 3D geometries; try a different --seed.")
        return tr_i, va_i, te_i

    def build_model():
        if args.model == "mace":
            if build_mace_for_geomo is None:
                raise RuntimeError("MACE requires: pip install mace-torch")
            return build_mace_for_geomo(
                input_dim=num_feats,
                hidden=args.hidden,
                n_layers=args.layers,
                dropout=args.dropout,
                num_tasks=num_tasks,
            ).to(device)
        return EnhancedOG_PGAT(
            input_dim=num_feats,
            hidden=args.hidden,
            n_layers=args.layers,
            dropout=args.dropout,
            num_tasks=num_tasks,
            edge_dim=num_edge_feats,
            disable_crf=False,
            mean_pool_only=False,
        ).to(device)

    def make_loaders(tr_i, va_i, te_i):
        trL = DataLoader([dataset_sample_with_geom(dataset, i) for i in tr_i], batch_size=args.batch, shuffle=True)
        vaL = DataLoader([dataset_sample_with_geom(dataset, i) for i in va_i], batch_size=args.batch)
        teL = DataLoader([dataset_sample_with_geom(dataset, i) for i in te_i], batch_size=args.batch)
        return trL, vaL, teL

    if args.eval_only:
        set_seed(args.seed)
        tr_i, va_i, te_i = make_split(args.seed)
        print(f"Split: Train={len(tr_i)}, Val={len(va_i)}, Test={len(te_i)} | Tasks={num_tasks}")
        _, vaL, teL = make_loaders(tr_i, va_i, te_i)
        model = build_model()
        if ckpt is not None:
            state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded weights from {ckpt_path}")
            if missing:
                print(f"  missing keys: {len(missing)}")
            if unexpected:
                print(f"  unexpected keys: {len(unexpected)}")
        val_auc = eval_epoch(model, vaL, device, num_tasks, 0.0)
        test_auc = eval_epoch(model, teL, device, num_tasks, 0.0)
        print(f"Eval-only [{args.dataset}] Val AUC={val_auc:.4f} | Test AUC={test_auc:.4f}")
        if ckpt is not None:
            print(f"Checkpoint recorded best Val AUC={ckpt.get('best_val_auc')} "
                  f"| Test AUC={ckpt.get('test_auc')}")
        return

    all_y = torch.cat([d.y.view(1, -1) for d in dataset], 0)
    all_y[all_y == -1] = float("nan")
    pos_w = []
    for i in range(all_y.shape[1]):
        yi = all_y[:, i]
        y0, y1 = torch.sum(yi == 0), torch.sum(yi == 1)
        pos_w.append((y0 / (y1 + 1e-6)).item() if y1 > 0 else 1.0)
    pos_weight = torch.tensor(pos_w, device=device)
    print("→ pos_weight:", pos_weight.cpu().numpy())

    n_runs = max(1, int(args.n_runs))
    run_val_aucs, run_test_aucs = [], []

    for run in range(n_runs):
        run_seed = args.seed + run
        set_seed(run_seed)
        print(f"\n===== Run {run + 1}/{n_runs} | seed={run_seed} =====")

        tr_i, va_i, te_i = make_split(run_seed)
        print(f"Split: Train={len(tr_i)}, Val={len(va_i)}, Test={len(te_i)} | Tasks={num_tasks}")
        trL, vaL, teL = make_loaders(tr_i, va_i, te_i)
        model = build_model()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)

        best_auc, best_state, bad_epochs = 0.0, None, 0
        loss_history, val_history = [], []

        def save_checkpoint(path, epoch, extra=None):
            payload = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "best_val_auc": best_auc,
                "seed": run_seed,
                "args": vars(args),
            }
            if extra:
                payload.update(extra)
            torch.save(payload, path)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            tr_loss = train_epoch(model, trL, opt, device, args.lambda_reason, pos_weight, 0.0)
            val_auc = eval_epoch(model, vaL, device, num_tasks, 0.0)
            sched.step()
            end_time = time.time()
            loss_history.append(tr_loss)
            val_history.append(val_auc)
            print(f"[{epoch:03d}] Loss={tr_loss:.4f} | Val AUC={val_auc:.4f} | Time Cost={(end_time-start_time):.4f}s")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
                checkpoint_path = os.path.join(
                    args.save_dir, f"{args.dataset}_seed{run_seed}_best_model.pth"
                )
                save_checkpoint(checkpoint_path, epoch)
                print(f"Best model saved to {checkpoint_path} (Val AUC={best_auc:.4f})")
            else:
                bad_epochs += 1

            if args.save_every > 0 and epoch % args.save_every == 0:
                periodic_path = os.path.join(
                    args.save_dir, f"{args.dataset}_seed{run_seed}_epoch{epoch:03d}.pth"
                )
                save_checkpoint(periodic_path, epoch)
                print(f"Periodic checkpoint saved to {periodic_path}")

            if bad_epochs >= args.patience:
                print("Early stopping: no improvement.")
                break

        last_path = os.path.join(args.save_dir, f"{args.dataset}_seed{run_seed}_last_model.pth")
        save_checkpoint(last_path, epoch)
        print(f"Last-epoch checkpoint saved to {last_path}")

        if best_state:
            model.load_state_dict(best_state)

        test_auc = eval_epoch(model, teL, device, num_tasks, 0.0)
        print(f"Run {run + 1} Test AUC = {test_auc:.4f} (best Val = {best_auc:.4f})")

        best_path = os.path.join(args.save_dir, f"{args.dataset}_seed{run_seed}_best_model.pth")
        if os.path.isfile(best_path):
            run_ckpt = torch.load(best_path, map_location="cpu")
            run_ckpt["test_auc"] = test_auc
            torch.save(run_ckpt, best_path)

        run_val_aucs.append(best_auc)
        run_test_aucs.append(test_auc)

        loss_conv = find_convergence_epoch(loss_history)
        val_conv = find_convergence_epoch(val_history)
        print(
            f"Convergence: Loss @ epoch {loss_conv if loss_conv else 'N/A'}, "
            f"Val AUC @ epoch {val_conv if val_conv else 'N/A'} "
            f"(total {len(loss_history)} epochs)"
        )

    val_arr = np.array(run_val_aucs, dtype=float)
    test_arr = np.array(run_test_aucs, dtype=float)
    val_std = float(val_arr.std(ddof=1)) if len(val_arr) > 1 else 0.0
    test_std = float(test_arr.std(ddof=1)) if len(test_arr) > 1 else 0.0
    print("\n" + "=" * 50)
    print(f"{args.dataset.upper()} over {n_runs} runs (seeds {args.seed}..{args.seed + n_runs - 1})")
    print(f"Val AUC:  {val_arr.mean():.4f} ± {val_std:.4f}  values={run_val_aucs}")
    print(f"Test AUC: {test_arr.mean():.4f} ± {test_std:.4f}  values={run_test_aucs}")
    print("=" * 50)

if __name__ == "__main__":

    main()
