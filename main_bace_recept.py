

import os

import argparse

from utils.config_utils import parse_args_with_gat_config

import random

import time

import numpy as np

import scipy.sparse as sp

from scipy.sparse.csgraph import shortest_path

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

def compute_rbf_features(pos, edge_index, num_rbf_centers=16, rbf_min=0.0, rbf_max=6.0):

    """计算边上两个原子在 3D 空间中的 RBF 距离特征"""

    row, col = edge_index

    dist = torch.norm(pos[row] - pos[col], p=2, dim=-1)

    centers = torch.linspace(rbf_min, rbf_max, num_rbf_centers)

    gamma = 1.0 / (centers[1] - centers[0])

    rbf_features = torch.exp(-gamma * (dist.unsqueeze(-1) - centers)**2)

    return rbf_features

def attach_positions(dataset, attach_z_for_mace: bool = True):

    """Cache pos/z/rbf per index (PyG dataset[i] often drops pos)."""

    dataset._geo_pos = {}

    dataset._geo_z = {}

    dataset._geo_rbf = {}

    missing, total = 0, len(dataset)

    for i in range(total):

        data = dataset[i]

        smi = getattr(data, "smiles", None)

        nh = data.x.shape[0]

        pos = smiles_to_3d_pos(smi, num_heavy_atoms=nh) if smi else None

        if pos is not None:

            dataset._geo_pos[i] = pos

            if hasattr(data, 'edge_index'):

                dataset._geo_rbf[i] = compute_rbf_features(pos, data.edge_index)

        else:

            missing += 1

        if attach_z_for_mace and smi is not None:

            z = smiles_to_atomic_numbers(smi, num_heavy_atoms=nh)

            if z is not None:

                dataset._geo_z[i] = z

    print(f"Added 3D coordinates & RBF features to {total - missing}/{total} molecules ({missing} without valid conformer).")

    return dataset

def dataset_sample_with_geom(dataset, idx):

    d = dataset[idx].clone()

    g_pos = getattr(dataset, "_geo_pos", None)

    g_z = getattr(dataset, "_geo_z", None)

    g_rbf = getattr(dataset, "_geo_rbf", None)

    if g_pos is not None and idx in g_pos:

        d.pos = g_pos[idx]

    if g_z is not None and idx in g_z:

        d.z = g_z[idx]

    if hasattr(d, 'edge_index') and d.edge_index is not None:

        num_edges = d.edge_index.shape[1]

        if g_rbf is not None and idx in g_rbf:

            rbf_feat = g_rbf[idx]

        else:

            rbf_feat = torch.zeros((num_edges, 16), dtype=torch.float)

        if hasattr(d, 'edge_attr') and d.edge_attr is not None:

            orig_edge_attr = d.edge_attr.float()

            if orig_edge_attr.dim() == 1:

                orig_edge_attr = orig_edge_attr.view(-1, 1)

        else:

            orig_edge_attr = torch.zeros((num_edges, 1), dtype=torch.float)

        d.edge_attr = torch.cat([orig_edge_attr, rbf_feat], dim=-1)

    return d

def reasoning_loss(model):

    """Safely extract curvature regularization term."""

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

def compute_mean_receptive_depth(model, loader, device):

    model.eval()

    receptive_depths = []

    for batch in loader:

        batch = batch.to(device)

        batch.x = batch.x.float().requires_grad_(True)

        out, _ = model(batch)

        if out.dim() > 1 and batch.y.dim() > 1:

            if out.shape[1] > batch.y.shape[1]:

                out = out[:, :batch.y.shape[1]]

        score = out.sum()

        model.zero_grad()

        score.backward()

        node_grads = batch.x.grad.norm(dim=-1).cpu().numpy()

        ptr = batch.ptr.cpu().numpy() if hasattr(batch, 'ptr') else None

        if ptr is None:

            _, counts = torch.unique(batch.batch, return_counts=True)

            ptr = np.cumsum(np.insert(counts.cpu().numpy(), 0, 0))

        edge_index = batch.edge_index.cpu().numpy()

        for i in range(len(ptr) - 1):

            start, end = ptr[i], ptr[i+1]

            num_nodes = end - start

            if num_nodes <= 1:

                continue

            mask = (edge_index[0] >= start) & (edge_index[0] < end)

            sub_edges = edge_index[:, mask] - start

            adj = sp.coo_matrix((np.ones(sub_edges.shape[1]), (sub_edges[0], sub_edges[1])), shape=(num_nodes, num_nodes))

            dist_matrix = shortest_path(csgraph=adj, directed=False, unweighted=True)

            dist_matrix[np.isinf(dist_matrix)] = 0

            g_weights = node_grads[start:end]

            sum_g = np.sum(g_weights)

            if sum_g > 1e-6:

                g_weights = g_weights / sum_g

                expected_depth = np.sum(np.outer(g_weights, g_weights) * dist_matrix)

                receptive_depths.append(expected_depth)

    if len(receptive_depths) == 0:

        return 0.0, 0.0

    mean_depth = np.mean(receptive_depths)

    std_depth = np.std(receptive_depths)

    return mean_depth, std_depth

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

        curv_pen = laplace_beltrami_penalty(kappa, getattr(batch, "edge_index", None))

        if not torch.isfinite(curv_pen):

            curv_pen = torch.tensor(0.0, device=device)

        loss = loss_main + lam * (reasoning_loss(model) + curv_pen)

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

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch", type=int, default=64)

    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--layers", type=int, default=20)

    parser.add_argument("--dropout", type=float, default=0.25)

    parser.add_argument("--lambda_reason", type=float, default=0.002)

    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--model", choices=("geomo", "mace"), default="geomo")

    args = parse_args_with_gat_config(parser)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    dataset = MoleculeNet(root=f"data/{args.dataset}", name=args.dataset)

    print(f"Loaded {args.dataset.upper()} with {len(dataset)} molecules")

    dataset = attach_positions(dataset)

    num_feats, num_tasks = dataset.num_node_features, dataset.num_classes or 1

    num_tasks = max(num_tasks, 1)

    edge_dim_2d = dataset.num_edge_features if dataset.num_edge_features > 0 else 1

    total_edge_dim = edge_dim_2d + 16

    tr_i, va_i, te_i = scaffold_split(dataset)

    print(f"Split: Train={len(tr_i)}, Val={len(va_i)}, Test={len(te_i)}, Tasks={num_tasks}")

    if args.model == "mace":

        has_geo = set(getattr(dataset, "_geo_pos", {}).keys())

        if len(has_geo) < len(dataset):

            tr_i = [i for i in tr_i if i in has_geo]

            va_i = [i for i in va_i if i in has_geo]

            te_i = [i for i in te_i if i in has_geo]

            if not tr_i or not va_i or not te_i:

                raise RuntimeError("MACE: a split is empty after geometry filter; try --seed.")

            print(f"MACE: splits with 3D → Train={len(tr_i)}, Val={len(va_i)}, Test={len(te_i)}")

    trL = DataLoader([dataset_sample_with_geom(dataset, i) for i in tr_i], batch_size=args.batch, shuffle=True)

    vaL = DataLoader([dataset_sample_with_geom(dataset, i) for i in va_i], batch_size=args.batch)

    teL = DataLoader([dataset_sample_with_geom(dataset, i) for i in te_i], batch_size=args.batch)

    if args.model == "mace":

        if build_mace_for_geomo is None:

            raise RuntimeError("MACE 需要: pip install mace-torch")

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

            edge_dim=total_edge_dim,

            disable_crf=False,

            mean_pool_only=False,

            use_se3=True,

        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr/20)

    all_y = torch.cat([d.y.view(1, -1) for d in dataset], 0)

    all_y[all_y == -1] = float("nan")

    pos_w = []

    for i in range(all_y.shape[1]):

        yi = all_y[:, i]

        y0, y1 = torch.sum(yi == 0), torch.sum(yi == 1)

        pos_w.append((y0 / (y1 + 1e-6)).item() if y1 > 0 else 1.0)

    pos_weight = torch.tensor(pos_w, device=device)

    print("→ pos_weight:", pos_weight.cpu().numpy())

    best_auc, best_state, bad_epochs = 0.0, None, 0

    loss_history, val_history = [], []

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

            best_auc, best_state, bad_epochs = val_auc, model.state_dict(), 0

        else:

            bad_epochs += 1

        if bad_epochs >= args.patience:

            print("Early stopping: no improvement.")

            break

    if best_state:

        model.load_state_dict(best_state)

    test_auc = eval_epoch(model, teL, device, num_tasks, 0.0)

    print(f"Final Test AUC = {test_auc:.4f} (best Val = {best_auc:.4f})")

    print("Computing Mean Receptive Depth on Test Set...")

    mean_depth, std_depth = compute_mean_receptive_depth(model, teL, device)

    print(f"Mean Receptive Depth: {mean_depth:.2f} ± {std_depth:.2f}")

    loss_conv = find_convergence_epoch(loss_history)

    val_conv = find_convergence_epoch(val_history)

    print(f"Convergence: Loss @ epoch {loss_conv if loss_conv else 'N/A'}, "

          f"Val AUC @ epoch {val_conv if val_conv else 'N/A'} "

          f"(total {len(loss_history)} epochs)")

if __name__ == "__main__":

    main()
