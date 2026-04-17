import os, argparse, random, numpy as np, time

from utils.config_utils import parse_args_with_gat_config

from collections import defaultdict

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedShuffleSplit

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet

from torch_geometric.loader import DataLoader

try:

    from rdkit import Chem

    from rdkit.Chem import Descriptors, AllChem

    from rdkit.Chem.Scaffolds import MurckoScaffold

    RD_AVAILABLE = True

except Exception:

    RD_AVAILABLE = False

    print("RDKit not found")

from model.sdn import EnhancedOG_PGAT, laplace_beltrami_penalty

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

def find_convergence_epoch(values, window=10, rel_threshold=0.05):

    for start in range(len(values) - window):

        ref = values[start]

        if abs(ref) < 1e-10:

            continue

        segment = values[start:start + window]

        max_rel_change = max(abs(v - ref) / abs(ref) for v in segment)

        if max_rel_change < rel_threshold:

            return start + 1

    return None

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

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

        mol = Chem.MolFromSmiles(smi) if RD_AVAILABLE and smi else None

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

        print("Stratified fallback split.")

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

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    n_heavy = mol.GetNumAtoms()

    mol_h = Chem.AddHs(mol)

    try:

        res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

        if res != 0 or mol_h.GetNumConformers() == 0:

            return None

        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)

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

def attach_positions(dataset, attach_z_for_mace: bool = False):

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

        for attr in ['edge_attr', 'edge_attr_2d', 'edge_attr_3d']:

            if hasattr(data, attr):

                delattr(data, attr)

        num_edges = data.edge_index.size(1)

        data.edge_attr = torch.zeros((num_edges, 3), dtype=torch.float)

    print(f"3D coords: {total - missing}/{total} molecules ({missing} failed).")

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

def prepare_batch_for_model(batch):

    device = batch.x.device

    num_nodes = batch.x.size(0)

    num_edges = batch.edge_index.size(1)

    if not hasattr(batch, 'pos') or batch.pos is None:

        batch.pos = torch.zeros((num_nodes, 3), device=device, dtype=torch.float)

    for attr in ['edge_attr', 'edge_attr_2d', 'edge_attr_3d']:

        if hasattr(batch, attr):

            delattr(batch, attr)

    batch.edge_attr = torch.zeros((num_edges, 3), device=device, dtype=torch.float)

    return batch

class FocalLoss(nn.Module):

    """
    Focal Loss for extreme class imbalance.
    alpha: weight for positive class (0.25 means 3:1 neg:pos weight)
    gamma: focusing parameter (higher = more focus on hard examples)
    """

    def __init__(self, alpha=0.25, gamma=2.0):

        super().__init__()

        self.alpha = alpha

        self.gamma = gamma

    def forward(self, logits, targets):

        """
        Args:
            logits: (N, C) raw scores
            targets: (N, C) binary labels
        Returns:
            (N, C) focal loss per element (NOT reduced)
        """

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits)

        pt = torch.where(targets > 0.5, probs, 1 - probs)

        alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)

        focal_weight = alpha_t * (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce

        return focal_loss

def train_epoch(model, loader, optimizer, device, lam, pos_weight=None,

                use_focal=False, noise_level=0.0):

    model.train()

    if use_focal:

        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        print("🎯 Using Focal Loss")

    else:

        criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_loss, steps = 0.0, 0

    for batch in loader:

        try:

            batch = batch.to(device)

            batch = prepare_batch_for_model(batch)

            x = batch.x.float()

            if noise_level > 0:

                x = x + noise_level * torch.randn_like(x)

            optimizer.zero_grad(set_to_none=True)

            batch_for_model = batch.clone() if hasattr(batch, 'clone') else batch

            batch_for_model.x = x

            out, kappa = model(batch_for_model)

            out = torch.clamp(out.float(), -20.0, 20.0)

            y = batch.y.float().view(out.size(0), -1)

            m = min(out.shape[1], y.shape[1])

            out, y = out[:, :m], y[:, :m]

            mask = (y != -1) & torch.isfinite(y)

            if mask.sum() == 0:

                continue

            loss_mat = criterion(out, y.nan_to_num(0.0))

            if pos_weight is not None and not use_focal:

                w = pos_weight.view(1, -1).expand_as(y)

                loss_mat = loss_mat * torch.where(y > 0.5, w, 1.0)

            loss_main = loss_mat[mask].mean()

            curv_pen = laplace_beltrami_penalty(kappa, getattr(batch, "edge_index", None))

            curv_pen = curv_pen if torch.isfinite(curv_pen) else torch.tensor(0.0, device=device)

            reg_loss = model.get_curvature_reg() if hasattr(model, 'get_curvature_reg') else 0.0

            loss = loss_main + lam * (reg_loss + curv_pen)

            if not torch.isfinite(loss):

                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += float(loss.detach().cpu())

            steps += 1

        except RuntimeError as e:

            print(f"RuntimeError: {e}")

            continue

    return total_loss / max(steps, 1)

@torch.no_grad()

def eval_epoch(model, loader, device, num_tasks, noise_level=0.0):

    model.eval()

    y_true, y_pred = [], []

    for batch in loader:

        try:

            batch = batch.to(device)

            batch = prepare_batch_for_model(batch)

            x = batch.x.float()

            if noise_level > 0:

                x = x + noise_level * torch.randn_like(x)

            batch_for_model = batch.clone() if hasattr(batch, 'clone') else batch

            batch_for_model.x = x

            out, _ = model(batch_for_model)

            out = out.float().clamp(-20, 20)

            y = batch.y.float().view(out.size(0), -1)

            m = min(out.shape[1], y.shape[1])

            out, y = out[:, :m], y[:, :m]

            mask = (y != -1) & torch.isfinite(y)

            if mask.sum() == 0:

                continue

            y_true.append(y[mask].cpu())

            y_pred.append(torch.sigmoid(out[mask]).cpu())

        except RuntimeError:

            continue

    if not y_true:

        return 0.0

    y_true = torch.cat(y_true, 0).numpy()

    y_pred = torch.cat(y_pred, 0).numpy()

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

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="hiv")

    ap.add_argument("--epochs", type=int, default=300)

    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--lr", type=float, default=1e-3)

    ap.add_argument("--hidden", type=int, default=384)

    ap.add_argument("--layers", type=int, default=8)

    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--lambda_reason", type=float, default=0.0005)

    ap.add_argument("--patience", type=int, default=40)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_focal", action="store_true")

    ap.add_argument("--weight_decay", type=float, default=5e-4)

    ap.add_argument(

        "--model",

        choices=("geomo", "mace"),

        default="geomo",

        help="geomo=EnhancedOG_PGAT, mace=MACE baseline (requires pip install mace-torch)",

    )

    ap.add_argument(

        "--prism_ablation",

        type=str,

        default="full",

        choices=("full", "random_scalar", "attention", "depth_embedding", "uniform"),

        help="Table (B) topology mixing; only for --model geomo.",

    )

    ap.add_argument("--n_runs", type=int, default=1, help="Repeat with seeds seed, seed+1, ...")

    args = parse_args_with_gat_config(ap)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    dataset = MoleculeNet(root=f"data/{args.dataset}", name=args.dataset)

    print(f"Loaded {args.dataset.upper()}: {len(dataset)} molecules")

    dataset = attach_positions(dataset, attach_z_for_mace=(args.model == "mace"))

    num_feats = dataset.num_node_features

    num_tasks = dataset.num_classes or 1

    num_tasks = max(num_tasks, 1)

    print(f"Features={num_feats}, Tasks={num_tasks}")

    all_y = []

    for d in dataset:

        y = d.y.view(-1)

        y = y[y != -1]

        all_y.append(y)

    all_y = torch.cat(all_y, 0)

    y0 = torch.sum(all_y == 0).item()

    y1 = torch.sum(all_y == 1).item()

    pos_weight_value = (y0 / (y1 + 1e-6)) if y1 > 0 else 1.0

    pos_weight = torch.tensor([pos_weight_value], device=device)

    print(f"→ Class distribution: Neg={y0}, Pos={y1}")

    print(f"→ pos_weight={pos_weight_value:.2f}")

    n_runs = max(1, int(args.n_runs))

    run_aucs = []

    for run in range(n_runs):

        run_seed = args.seed + run

        set_seed(run_seed)

        if n_runs > 1:

            print(f"\n===== Run {run + 1}/{n_runs} | seed={run_seed} =====")

        tr_idx, val_idx, te_idx = scaffold_split(dataset, seed=run_seed)

        print(f"Split: Train={len(tr_idx)}, Val={len(val_idx)}, Test={len(te_idx)}")

        has_geo = set(getattr(dataset, "_geo_pos", {}).keys())

        if len(has_geo) < len(dataset):

            tr_idx = [i for i in tr_idx if i in has_geo]

            val_idx = [i for i in val_idx if i in has_geo]

            te_idx = [i for i in te_idx if i in has_geo]

            print(f"Filtered → Train={len(tr_idx)}, Val={len(val_idx)}, Test={len(te_idx)}")

        if args.model == "mace":

            has_z = set(getattr(dataset, "_geo_z", {}).keys())

            tr_idx = [i for i in tr_idx if i in has_z]

            val_idx = [i for i in val_idx if i in has_z]

            te_idx = [i for i in te_idx if i in has_z]

            if not tr_idx or not val_idx or not te_idx:

                raise RuntimeError(

                    "MACE: a split is empty after requiring cached atomic numbers (_geo_z); try --seed."

                )

            print(f"MACE: filtered to samples with z → Train={len(tr_idx)}, Val={len(val_idx)}, Test={len(te_idx)}")

        trL = DataLoader(

            [dataset_sample_with_geom(dataset, i) for i in tr_idx],

            batch_size=args.batch,

            shuffle=True,

        )

        vaL = DataLoader(

            [dataset_sample_with_geom(dataset, i) for i in val_idx],

            batch_size=args.batch,

        )

        teL = DataLoader(

            [dataset_sample_with_geom(dataset, i) for i in te_idx],

            batch_size=args.batch,

        )

        if args.model == "mace":

            if build_mace_for_geomo is None:

                raise RuntimeError("MACE requires: pip install mace-torch")

            model = build_mace_for_geomo(

                input_dim=num_feats,

                hidden=args.hidden,

                n_layers=args.layers,

                dropout=args.dropout,

                num_tasks=num_tasks,

            ).to(device)

        else:

            model = EnhancedOG_PGAT(

                input_dim=num_feats,

                hidden=args.hidden,

                n_layers=args.layers,

                dropout=args.dropout,

                num_tasks=num_tasks,

                edge_dim=3,

                disable_crf=False,

                mean_pool_only=False,

                prism_ablation=args.prism_ablation,

            ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)

        best_auc, best_state, bad_epochs = 0.0, None, 0

        loss_history, val_history = [], []

        for epoch in range(1, args.epochs + 1):

            start_time = time.time()

            tr_loss = train_epoch(

                model,

                trL,

                opt,

                device,

                args.lambda_reason,

                pos_weight if not args.use_focal else None,

                args.use_focal,

                0.01,

            )

            val_auc = eval_epoch(model, vaL, device, num_tasks, 0.0)

            sched.step()

            end_time = time.time()

            loss_history.append(tr_loss)

            val_history.append(val_auc)

            status = "*" if val_auc > best_auc else "  "

            print(

                f"{status}[{epoch:03d}] Loss={tr_loss:.4f} | Val AUC={val_auc:.4f} | "

                f"Time={(end_time-start_time):.2f}s"

            )

            if val_auc > best_auc:

                best_auc = val_auc

                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

                bad_epochs = 0

            else:

                bad_epochs += 1

            if bad_epochs >= args.patience:

                print(f"Early stop @ epoch {epoch}")

                break

        if best_state:

            model.load_state_dict(best_state)

        test_auc = eval_epoch(model, teL, device, num_tasks, 0.0)

        print(f"\nFinal Test AUC = {test_auc:.4f} (best Val = {best_auc:.4f})")

        run_aucs.append(float(test_auc))

        loss_conv = find_convergence_epoch(loss_history)

        val_conv = find_convergence_epoch(val_history)

        print(

            f"Convergence: Loss @ {loss_conv or 'N/A'}, Val @ {val_conv or 'N/A'} "

            f"({len(loss_history)} epochs)"

        )

    if n_runs > 1 and run_aucs:

        a = np.asarray(run_aucs, dtype=float)

        a_std = float(a.std(ddof=1)) if len(a) > 1 else 0.0

        print(f"\nTest AUC over {n_runs} runs: mean={a.mean():.4f}, std={a_std:.4f}, values={run_aucs}")

if __name__ == "__main__":

    main()
