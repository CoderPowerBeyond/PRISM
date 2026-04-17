

import os

import argparse

from utils.config_utils import parse_args_with_gat_config

import random

import numpy as np

from collections import defaultdict

from rdkit import Chem

from rdkit.Chem import AllChem, Descriptors

from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import mean_squared_error

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet

from torch_geometric.loader import DataLoader

import model.sdn_gcn as m_gcn

from model.gcn_pure import PureGCN

import model.gin as m_gin

from model.gin_pure import PureGIN

import model.gat_baseline as m_gat

from model.gat_pure import PureGAT

from model.equformer import EnhancedOG_PGAT, laplace_beltrami_penalty

try:

    from model.mace import build_mace_for_geomo

except ImportError:

    build_mace_for_geomo = None

import time

def build_model(args, num_feats, num_tasks, device):

    """GeoMo (ours) or MACE / GCN / GIN / GAT baselines."""

    if args.model == "mace":

        if build_mace_for_geomo is None:

            raise RuntimeError("MACE not installed; pip install mace-torch or use --model geomo|gcn|gin|gat")

        return build_mace_for_geomo(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gcn":

        return PureGCN(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gcn_sdn":

        return m_gcn.EnhancedOG_PGAT(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            mean_pool_only=False,

        ).to(device)

    if args.model == "gin":

        return PureGIN(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gin_sdn":

        return m_gin.EnhancedOG_PGAT(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            disable_crf=False,

            mean_pool_only=False,

        ).to(device)

    if args.model == "gat":

        return PureGAT(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gat_sdn":

        return m_gat.EnhancedOG_PGAT(

            input_dim=num_feats,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            mean_pool_only=False,

        ).to(device)

    return EnhancedOG_PGAT(

        input_dim=num_feats,

        hidden=args.hidden,

        n_layers=args.layers,

        dropout=args.dropout,

        num_tasks=num_tasks,

        disable_crf=False,

        mean_pool_only=False,

        use_se3=True,

        prism_ablation=getattr(args, "prism_ablation", "full"),

    ).to(device)

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

    """Bemis–Murcko scaffold split for regression."""

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

    n_total = len(dataset)

    n_train, n_val = int(frac_train * n_total), int(frac_val * n_total)

    train, val, test = [], [], []

    for g in groups:

        if len(train) + len(g) <= n_train:

            train += g

        elif len(val) + len(g) <= n_val:

            val += g

        else:

            test += g

    if not val or not test:

        return random_split(dataset, frac_train, frac_val, frac_test, seed)

    return train, val, test

def random_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

    """Random split (Standard for ESOL to achieve ~0.6 RMSE)."""

    all_idx = np.arange(len(dataset))

    np.random.seed(seed)

    np.random.shuffle(all_idx)

    n_train, n_val = int(frac_train*len(all_idx)), int(frac_val*len(all_idx))

    train = all_idx[:n_train].tolist()

    val = all_idx[n_train:n_train+n_val].tolist()

    test = all_idx[n_train+n_val:].tolist()

    return train, val, test

def smiles_to_3d_pos(smiles: str, num_heavy_atoms: int = None):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    n_heavy = mol.GetNumAtoms()

    mol_h = Chem.AddHs(mol)

    try:

        AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())

        AllChem.UFFOptimizeMolecule(mol_h)

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

    z_list = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_heavy)]

    z = torch.tensor(z_list, dtype=torch.long)

    if num_heavy_atoms is not None and z.shape[0] != num_heavy_atoms:

        return None

    return z

def attach_positions(dataset, attach_z_for_mace: bool = True):

    dataset._geo_pos = {}

    dataset._geo_z = {}

    missing = 0

    for i in range(len(dataset)):

        data = dataset[i]

        smi = getattr(data, "smiles", None)

        n_heavy = data.x.shape[0]

        pos = smiles_to_3d_pos(smi, num_heavy_atoms=n_heavy) if smi else None

        if pos is not None:

            dataset._geo_pos[i] = pos

        else:

            missing += 1

        if attach_z_for_mace and smi is not None:

            z = smiles_to_atomic_numbers(smi, num_heavy_atoms=n_heavy)

            if z is not None:

                dataset._geo_z[i] = z

    print(f"Attached 3D coordinates: {len(dataset)-missing}/{len(dataset)} molecules.")

    return dataset

def dataset_sample_with_geom(dataset, idx):

    d = dataset[idx].clone()

    g = getattr(dataset, "_geo_pos", None)

    if g is not None and idx in g:

        d.pos = g[idx]

    elif getattr(d, "pos", None) is None:

        d.pos = torch.zeros((d.x.shape[0], 3), dtype=torch.float)

    gz = getattr(dataset, "_geo_z", None)

    if gz is not None and idx in gz:

        d.z = gz[idx]

    return d

def reasoning_loss(model):

    try:

        return model.get_curvature_reg()

    except Exception:

        return torch.tensor(0.0, device=next(model.parameters()).device)

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

def train_epoch(model, loader, optimizer, device, lam, noise_level=0.0, y_mean=0.0, y_std=1.0):

    model.train()

    total_loss, k = 0.0, 0

    mse = nn.MSELoss(reduction="none")

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn(batch.x.shape, device=batch.x.device)

        optimizer.zero_grad()

        out, kappa = model(batch)

        out = out.view(-1)

        y = batch.y.view(-1).float()

        y_norm = (y - y_mean) / y_std

        mask = torch.isfinite(y_norm)

        if mask.sum() == 0:

            continue

        loss_main = mse(out[mask], y_norm[mask]).mean()

        curv_pen = laplace_beltrami_penalty(kappa, getattr(batch, "edge_index", None))

        curv_pen = curv_pen if torch.isfinite(curv_pen) else torch.tensor(0.0, device=device)

        loss = loss_main + lam * (reasoning_loss(model) + curv_pen)

        if not torch.isfinite(loss):

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total_loss += float(loss)

        k += 1

    return total_loss / max(k, 1)

@torch.no_grad()

def eval_epoch(model, loader, device, noise_level=0.0, y_mean=0.0, y_std=1.0):

    model.eval()

    preds, trues = [], []

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn(batch.x.shape, device=batch.x.device)

        out, _ = model(batch)

        out_unnorm = out.view(-1).cpu() * y_std + y_mean

        preds.append(out_unnorm)

        trues.append(batch.y.view(-1).cpu())

    if not preds:

        return None

    pred, true = torch.cat(preds), torch.cat(trues)

    mask = torch.isfinite(true)

    pred, true = pred[mask], true[mask]

    if len(true) == 0:

        return None

    mse_val = mean_squared_error(true.numpy(), pred.numpy())

    rmse = float(np.sqrt(mse_val))

    mae = float(torch.mean(torch.abs(pred - true)))

    return rmse, mae

def main():

    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="esol", choices=("esol", "lipophilicity", "lipo"))

    ap.add_argument("--split", default="random", choices=("random", "scaffold"))

    ap.add_argument("--epochs", type=int, default=400)

    ap.add_argument("--batch", type=int, default=128)

    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--hidden", type=int, default=192)

    ap.add_argument("--layers", type=int, default=5)

    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lambda_reason", type=float, default=0.001)

    ap.add_argument("--patience", type=int, default=100)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(

        "--model",

        choices=("geomo", "mace", "gcn", "gcn_sdn", "gin", "gin_sdn", "gat", "gat_sdn"),

        default="gat",

    )

    ap.add_argument("--prism_ablation", type=str, default="full")

    ap.add_argument("--n_runs", type=int, default=1)

    args = parse_args_with_gat_config(ap)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    ds = args.dataset.lower()

    if ds in ("lipophilicity", "lipo"):

        root, name = "data/lipo", "lipo"

    else:

        root, name = "data/esol", "ESOL"

    dataset = MoleculeNet(root=root, name=name)

    dataset = attach_positions(dataset)

    if args.split == "scaffold":

        tr_i, va_i, te_i = scaffold_split(dataset, seed=args.seed)

    else:

        tr_i, va_i, te_i = random_split(dataset, seed=args.seed)

    num_feats, num_tasks = dataset.num_node_features, 1

    train_y = torch.cat([dataset[i].y for i in tr_i]).view(-1)

    y_mean, y_std = train_y.mean().item(), train_y.std().item() or 1.0

    run_rmses, run_maes = [], []

    for run in range(max(1, args.n_runs)):

        set_seed(args.seed + run)

        trL = DataLoader([dataset_sample_with_geom(dataset, i) for i in tr_i], batch_size=args.batch, shuffle=True)

        vaL = DataLoader([dataset_sample_with_geom(dataset, i) for i in va_i], batch_size=args.batch)

        teL = DataLoader([dataset_sample_with_geom(dataset, i) for i in te_i], batch_size=args.batch)

        model = build_model(args, num_feats, num_tasks, device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)

        best_rmse, best_state, bad = 9999.0, None, 0

        epoch_times = []

        for ep in range(1, args.epochs + 1):

            start_time = time.perf_counter()

            tr_loss = train_epoch(model, trL, opt, device, args.lambda_reason, 0.0, y_mean, y_std)

            val_eval = eval_epoch(model, vaL, device, 0.0, y_mean, y_std)

            sched.step()

            end_time = time.perf_counter()

            epoch_time = end_time - start_time

            epoch_times.append(epoch_time)

            avg_time = np.mean(epoch_times)

            print(f"[{ep:03d}] Epoch time: {epoch_time:.2f}s (avg {avg_time:.2f}s)")

            if val_eval is None:

                continue

            val_rmse, val_mae = val_eval

            if ep % 10 == 0 or ep == 1:

                print(f"     Loss={tr_loss:.4f} | Val RMSE={val_rmse:.4f} | MAE={val_mae:.4f}")

            if val_rmse < best_rmse:

                best_rmse, best_state, bad = val_rmse, model.state_dict(), 0

            else:

                bad += 1

            if bad >= args.patience:

                print(f"Early stopping at epoch {ep}")

                break

        if best_state:

            model.load_state_dict(best_state)

        test_eval = eval_epoch(model, teL, device, 0.0, y_mean, y_std)

        if test_eval:

            test_rmse, test_mae = test_eval

            run_rmses.append(float(test_rmse))

            run_maes.append(float(test_mae))

            print(f"Test RMSE = {test_rmse:.4f} | MAE = {test_mae:.4f} | Best Val RMSE = {best_rmse:.4f}")

            print(f"Average epoch time: {np.mean(epoch_times):.2f}s over {len(epoch_times)} epochs")

    if len(run_rmses) > 1:

        print(f"\nRMSEs over {len(run_rmses)} runs: mean={np.mean(run_rmses):.4f} ± {np.std(run_rmses):.4f}")

        print(f"MAEs over {len(run_maes)} runs: mean={np.mean(run_maes):.4f} ± {np.std(run_maes):.4f}")

if __name__ == "__main__":

    main()
