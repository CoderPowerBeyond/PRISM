

"""
main_muv.py — Optimized for MUV dataset (extreme imbalance, ~30 positives/task)
Target: Test AUC > 0.90
"""

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

    if len(values) < window:

        return None

    for start in range(len(values) - window):

        ref = values[start]

        if abs(ref) < 1e-10:

            continue

        segment = values[start:start + window]

        max_rel_change = max(abs(v - ref) / abs(ref) for v in segment)

        if max_rel_change < rel_threshold:

            return start + 1

    return None

def scaffold_split_multi_task(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

    """
    MUV-specific: ensure each split has positives for all tasks
    """

    n = len(dataset)

    all_labels = []

    for d in dataset:

        y = d.y.view(-1).numpy()

        all_labels.append(y)

    all_labels = np.array(all_labels)

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

    n_train = int(frac_train * n)

    n_val = int(frac_val * n)

    train, val, test = [], [], []

    train_pos = np.zeros(all_labels.shape[1])

    val_pos = np.zeros(all_labels.shape[1])

    test_pos = np.zeros(all_labels.shape[1])

    for g in groups:

        g_labels = all_labels[g]

        g_pos = np.sum(g_labels == 1, axis=0)

        if len(train) < n_train:

            train += g

            train_pos += g_pos

        elif len(val) < n_val:

            val += g

            val_pos += g_pos

        else:

            test += g

            test_pos += g_pos

    print("Per-split positives:")

    print(f"   Train: {train_pos.astype(int)}")

    print(f"   Val:   {val_pos.astype(int)}")

    print(f"   Test:  {test_pos.astype(int)}")

    if np.any(val_pos == 0) or np.any(test_pos == 0):

        print("Using stratified fallback (some tasks have no positives)")

        proxy_labels = all_labels[:, 0]

        proxy_labels[proxy_labels == -1] = 0

        X = np.zeros((len(proxy_labels), 1))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac_val + frac_test, random_state=seed)

        train_idx, temp_idx = next(sss.split(X, proxy_labels))

        tval = frac_val / (frac_val + frac_test)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tval, random_state=seed)

        val_idx, test_idx = next(sss2.split(X[temp_idx], proxy_labels[temp_idx]))

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

class TaskWeightedFocalLoss(nn.Module):

    """
    MUV-specific: dynamic per-task alpha based on class imbalance
    """

    def __init__(self, gamma=2.0):

        super().__init__()

        self.gamma = gamma

    def forward(self, logits, targets, task_alphas):

        """
        Args:
            logits: (N, num_tasks)
            targets: (N, num_tasks)
            task_alphas: (num_tasks,) per-task alpha values
        """

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        probs = torch.sigmoid(logits)

        pt = torch.where(targets > 0.5, probs, 1 - probs)

        alpha_t = torch.where(targets > 0.5,

                              task_alphas.view(1, -1),

                              1 - task_alphas.view(1, -1))

        focal_weight = alpha_t * (1 - pt) ** self.gamma

        focal_loss = focal_weight * bce

        return focal_loss

def mixup_graphs(batch1, batch2, alpha=0.2):

    """
    Mixup between two batches (at graph level)
    """

    lam = np.random.beta(alpha, alpha)

    batch1.x = lam * batch1.x + (1 - lam) * batch2.x

    if hasattr(batch1, 'pos') and hasattr(batch2, 'pos'):

        batch1.pos = lam * batch1.pos + (1 - lam) * batch2.pos

    batch1.y = lam * batch1.y + (1 - lam) * batch2.y

    return batch1

def train_epoch(model, loader, optimizer, device, lam, task_alphas,

                use_focal=True, use_mixup=True, noise_level=0.0):

    model.train()

    if use_focal:

        criterion = TaskWeightedFocalLoss(gamma=2.5)

    else:

        criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_loss, steps = 0.0, 0

    batch_list = list(loader)

    for idx, batch in enumerate(batch_list):

        try:

            batch = batch.to(device)

            batch = prepare_batch_for_model(batch)

            x = batch.x.float()

            if noise_level > 0:

                x = x + noise_level * torch.randn_like(x)

            if use_mixup and random.random() < 0.5 and idx < len(batch_list) - 1:

                batch2 = batch_list[idx + 1].to(device)

                batch2 = prepare_batch_for_model(batch2)

                batch = mixup_graphs(batch, batch2, alpha=0.2)

            optimizer.zero_grad(set_to_none=True)

            batch.x = x

            out, kappa = model(batch)

            out = torch.clamp(out.float(), -20.0, 20.0)

            y = batch.y.float()

            if out.shape[1] != y.shape[1]:

                m = min(out.shape[1], y.shape[1])

                out, y = out[:, :m], y[:, :m]

            mask = (y != -1) & torch.isfinite(y)

            if mask.sum() == 0:

                continue

            if use_focal:

                loss_mat = criterion(out, y.nan_to_num(0.0), task_alphas)

            else:

                loss_mat = criterion(out, y.nan_to_num(0.0))

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

def eval_epoch(model, loader, device, num_tasks, tta_rounds=5):

    """
    Test-Time Augmentation: average predictions over multiple forward passes
    """

    model.eval()

    y_true, y_pred = [], []

    for batch in loader:

        try:

            batch = batch.to(device)

            batch = prepare_batch_for_model(batch)

            preds = []

            for _ in range(tta_rounds):

                x = batch.x.float()

                x = x + 0.01 * torch.randn_like(x)

                batch_aug = batch.clone() if hasattr(batch, 'clone') else batch

                batch_aug.x = x

                out, _ = model(batch_aug)

                out = out.float().clamp(-20, 20)

                preds.append(torch.sigmoid(out))

            out_mean = torch.stack(preds).mean(dim=0)

            y = batch.y.float()

            if out_mean.shape[1] != y.shape[1]:

                m = min(out_mean.shape[1], y.shape[1])

                out_mean, y = out_mean[:, :m], y[:, :m]

            mask = (y != -1) & torch.isfinite(y)

            if mask.sum() == 0:

                continue

            y_true.append(y[mask].cpu())

            y_pred.append(out_mean[mask].cpu())

        except RuntimeError:

            continue

    if not y_true:

        return 0.0

    y_true = torch.cat(y_true, 0).numpy()

    y_pred = torch.cat(y_pred, 0).numpy()

    def safe_auc(y, p):

        if len(np.unique(y)) < 2:

            return np.nan

        try:

            return roc_auc_score(y, p)

        except:

            return np.nan

    task_aucs = []

    if y_true.ndim == 1 or y_true.shape[1] == 1:

        y_true = y_true.reshape(-1)

        y_pred = y_pred.reshape(-1)

        a1 = safe_auc(y_true, y_pred)

        a2 = safe_auc(y_true, 1 - y_pred)

        return float(np.nanmax([a1, a2]))

    for i in range(y_true.shape[1]):

        yi = y_true[:, i]

        pi = y_pred[:, i]

        valid = ~np.isnan(yi)

        yi, pi = yi[valid], pi[valid]

        if len(np.unique(yi)) < 2:

            continue

        a1 = safe_auc(yi, pi)

        a2 = safe_auc(yi, 1 - pi)

        task_aucs.append(np.nanmax([a1, a2]))

    return float(np.nanmean(task_aucs)) if task_aucs else 0.0

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="muv")

    ap.add_argument("--epochs", type=int, default=400)

    ap.add_argument("--batch", type=int, default=64)

    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--hidden", type=int, default=512)

    ap.add_argument("--layers", type=int, default=10)

    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--lambda_reason", type=float, default=0.0001)

    ap.add_argument("--patience", type=int, default=60)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_mixup", action="store_true", default=True)

    ap.add_argument("--tta_rounds", type=int, default=5)

    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument(

        "--model",

        choices=("geomo", "mace"),

        default="geomo",

        help="geomo=EnhancedOG_PGAT, mace=MACE baseline (requires pip install mace-torch)",

    )

    args = parse_args_with_gat_config(ap)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    dataset = MoleculeNet(root=f"data/{args.dataset}", name=args.dataset)

    print(f"Loaded {args.dataset.upper()}: {len(dataset)} molecules")

    dataset = attach_positions(dataset, attach_z_for_mace=(args.model == "mace"))

    num_feats = dataset.num_node_features

    num_tasks = dataset.num_classes or 1

    print(f"Features={num_feats}, Tasks={num_tasks}")

    tr_idx, val_idx, te_idx = scaffold_split_multi_task(dataset, seed=args.seed)

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

    trL = DataLoader([dataset_sample_with_geom(dataset, i) for i in tr_idx],

                     batch_size=args.batch, shuffle=True)

    vaL = DataLoader([dataset_sample_with_geom(dataset, i) for i in val_idx],

                     batch_size=args.batch)

    teL = DataLoader([dataset_sample_with_geom(dataset, i) for i in te_idx],

                     batch_size=args.batch)

    all_y = []

    for d in dataset:

        all_y.append(d.y.view(1, -1))

    all_y = torch.cat(all_y, 0)

    task_alphas = []

    for i in range(all_y.shape[1]):

        yi = all_y[:, i]

        yi = yi[yi != -1]

        if len(yi) == 0:

            task_alphas.append(0.5)

            continue

        n_pos = torch.sum(yi == 1).item()

        n_neg = torch.sum(yi == 0).item()

        alpha = n_neg / (n_pos + n_neg + 1e-6)

        task_alphas.append(min(alpha, 0.95))

    task_alphas = torch.tensor(task_alphas, device=device)

    print(f"→ Task alphas: {task_alphas.cpu().numpy()}")

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

        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = 20

    def lr_lambda(epoch):

        if epoch < warmup_epochs:

            return (epoch + 1) / warmup_epochs

        else:

            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_auc, best_state, bad_epochs = 0.0, None, 0

    loss_history, val_history = [], []

    use_mixup = bool(args.use_mixup and args.model != "mace")

    if args.model == "mace" and args.use_mixup:

        print("Note: graph mixup is disabled for --model mace.")

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        tr_loss = train_epoch(

            model,

            trL,

            opt,

            device,

            args.lambda_reason,

            task_alphas,

            use_focal=True,

            use_mixup=use_mixup,

            noise_level=0.02,

        )

        tta = args.tta_rounds if epoch % 5 == 0 else 1

        val_auc = eval_epoch(model, vaL, device, num_tasks, tta_rounds=tta)

        sched.step()

        end_time = time.time()

        loss_history.append(tr_loss)

        val_history.append(val_auc)

        status = "*" if val_auc > best_auc else "  "

        print(f"{status}[{epoch:03d}] Loss={tr_loss:.4f} | Val AUC={val_auc:.4f} | "

              f"Time={(end_time-start_time):.2f}s")

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

    test_auc = eval_epoch(model, teL, device, num_tasks, tta_rounds=args.tta_rounds)

    print(f"\nFinal Test AUC = {test_auc:.4f} (best Val = {best_auc:.4f})")

    loss_conv = find_convergence_epoch(loss_history)

    val_conv = find_convergence_epoch(val_history)

    print(f"Convergence: Loss @ {loss_conv or 'N/A'}, Val @ {val_conv or 'N/A'} "

          f"({len(loss_history)} epochs)")

if __name__ == "__main__":

    main()
