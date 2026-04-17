

import os

import time

import random

import numpy as np

from collections import defaultdict

from rdkit import Chem

from rdkit.Chem import AllChem

from rdkit.Chem.Scaffolds import MurckoScaffold

from tqdm import tqdm

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.model_selection import StratifiedShuffleSplit

import torch

import torch.nn.functional as F

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from model.sdn import EnhancedOG_PGAT, laplace_beltrami_penalty

import model.sdn_gcn as m_gcn

from model.gcn_pure import PureGCN

import model.gin as m_gin

from model.gin_pure import PureGIN

import model.gat_baseline as m_gat

from model.gat_pure import PureGAT

from utils.config_utils import parse_args_with_gat_config

try:

    from model.mace import build_mace_for_geomo

except ImportError:

    build_mace_for_geomo = None

def build_model(args, input_dim, num_tasks, device):

    """geomo (ours) or gcn / gin / gat / mace baselines."""

    if args.model == "mace":

        if build_mace_for_geomo is None:

            raise RuntimeError("MACE not installed; use --model geomo|gcn|gin|gat")

        return build_mace_for_geomo(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gcn":

        return PureGCN(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gcn_sdn":

        return m_gcn.EnhancedOG_PGAT(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            mean_pool_only=False,

        ).to(device)

    if args.model == "gin":

        return PureGIN(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gin_sdn":

        return m_gin.EnhancedOG_PGAT(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            disable_crf=False,

            mean_pool_only=False,

        ).to(device)

    if args.model == "gat":

        return PureGAT(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

        ).to(device)

    if args.model == "gat_sdn":

        return m_gat.EnhancedOG_PGAT(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=num_tasks,

            mean_pool_only=False,

        ).to(device)

    return EnhancedOG_PGAT(

        input_dim=input_dim,

        hidden=args.hidden,

        n_layers=args.layers,

        dropout=args.dropout,

        num_tasks=num_tasks,

        disable_crf=False,

        mean_pool_only=False,

        use_se3=True,

        prism_ablation=getattr(args, "prism_ablation", "full"),

    ).to(device)

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

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def warmup_cosine_scheduler(optimizer, warmup, total):

    """Linear warm‑up then cosine decay."""

    def lr_lambda(step):

        if step < warmup:

            return float(step) / float(max(1, warmup))

        p = (step - warmup) / float(max(1, total - warmup))

        return 0.5 * (1.0 + np.cos(np.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def smiles_to_3d_pos(smiles: str, num_heavy_atoms: int = None):

    """Generate 3D coordinates for heavy atoms only (matching data.x rows)."""

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

def attach_positions(dataset):

    """Attach 3D coordinate tensor to each molecule in dataset."""

    missing, total = 0, len(dataset)

    for i, data in enumerate(dataset):

        if getattr(data, "pos", None) is None:

            smi = getattr(data, "smiles", None)

            pos = smiles_to_3d_pos(smi, num_heavy_atoms=data.x.shape[0]) if smi else None

            if pos is not None:

                data.pos = pos

            else:

                missing += 1

    print(

        f"Added 3D coordinates to {total - missing}/{total} molecules "

        f"({missing} without valid conformer)."

    )

    return dataset

def atom_features(atom):

    return torch.tensor([

        atom.GetAtomicNum(),

        atom.GetTotalDegree(),

        atom.GetFormalCharge(),

        atom.GetTotalNumHs(),

        int(atom.GetIsAromatic())

    ], dtype=torch.float)

def bond_features(bond):

    bt = float(bond.GetBondTypeAsDouble())

    in_ring = float(bond.IsInRing())

    is_conj = float(bond.GetIsConjugated())

    return torch.tensor([bt, in_ring, is_conj], dtype=torch.float)

def load_freesolv(csv_path, limit=None):

    import pandas as pd

    df = pd.read_csv(csv_path)

    data_list = []

    for smi, yval in tqdm(zip(df["smiles"], df["y"]), total=len(df)):

        mol = Chem.MolFromSmiles(smi)

        if mol is None:

            continue

        x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

        row, col, edge_attr = [], [], []

        for bond in mol.GetBonds():

            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            e = bond_features(bond)

            row += [i, j]

            col += [j, i]

            edge_attr += [e, e]

        edge_index = torch.tensor([row, col], dtype=torch.long) if row else torch.zeros((2, 0), dtype=torch.long)

        edge_attr = torch.stack(edge_attr) if edge_attr else torch.zeros((0, 3), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([float(yval)]), smiles=smi)

        data_list.append(data)

        if limit and len(data_list) >= limit:

            break

    print(f"Loaded {len(data_list)} molecules from {csv_path}")

    return data_list

def scaffold_split(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=42):

    """Split dataset by scaffold with stratified fallback."""

    labels = np.array([float(d.y.item()) for d in dataset])

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

    n = len(dataset)

    n_train, n_val = int(frac_train * n), int(frac_val * n)

    train, val, test = [], [], []

    for g in groups:

        if len(train) + len(g) <= n_train:

            train += g

        elif len(val) + len(g) <= n_val:

            val += g

        else:

            test += g

    print(f"🔀 Scaffold split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    if not train or not val or not test:

        print("Fallback to stratified split.")

        X = np.zeros((len(labels), 1))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac_val + frac_test, random_state=seed)

        tr_idx, tmp_idx = next(sss.split(X, labels > np.median(labels)))

        tval = frac_val / (frac_val + frac_test)

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=tval, random_state=seed)

        va_idx, te_idx = next(sss2.split(X[tmp_idx], labels[tmp_idx] > np.median(labels)))

        va_idx, te_idx = tmp_idx[va_idx], tmp_idx[te_idx]

        train, val, test = tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()

    return train, val, test

def prepare_label_normalization(train_ds, eval_datasets):

    train_y = torch.tensor([float(data.y.view(-1)[0]) for data in train_ds], dtype=torch.float)

    y_mean = float(train_y.mean())

    y_std = float(train_y.std(unbiased=False))

    if y_std < 1e-8:

        y_std = 1.0

    for dataset in [train_ds] + list(eval_datasets):

        for data in dataset:

            y_value = data.y.view(-1).float()

            data.y_norm = (y_value - y_mean) / y_std

    return y_mean, y_std

def train_epoch(model, loader, optimizer, device, lam=0.001, noise_level=0.0):

    model.train()

    total_loss, steps = 0.0, 0

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn_like(batch.x.float())

        optimizer.zero_grad()

        pred, kappa = model(batch)

        loss_main = F.mse_loss(pred.view(-1), batch.y_norm.view(-1))

        curv_pen = laplace_beltrami_penalty(kappa, getattr(batch, "edge_index", None))

        curv_pen = curv_pen if torch.isfinite(curv_pen) else torch.tensor(0.0, device=device)

        reg_loss = model.get_curvature_reg() if hasattr(model, 'get_curvature_reg') else 0.0

        loss = loss_main + lam * (reg_loss + curv_pen)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        steps += batch.num_graphs

    return total_loss / max(steps, 1)

@torch.no_grad()

def evaluate(model, loader, device, y_mean, y_std, noise_level=0.0):

    model.eval()

    preds, trues = [], []

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn_like(batch.x.float())

        pred, _ = model(batch)

        pred = pred.view(-1) * y_std + y_mean

        preds.append(pred.cpu())

        trues.append(batch.y.view(-1).cpu())

    if not preds:

        return float('inf'), float('inf')

    preds = torch.cat(preds).numpy()

    trues = torch.cat(trues).numpy()

    mask = np.isfinite(trues) & np.isfinite(preds)

    trues, preds = trues[mask], preds[mask]

    if len(trues) == 0:

        return float('inf'), float('inf')

    mae = mean_absolute_error(trues, preds)

    rmse = float(np.sqrt(mean_squared_error(trues, preds)))

    return mae, rmse

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", default="data/freesolv.csv")

    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--batch", type=int, default=32)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--layers", type=int, default=6)

    parser.add_argument("--dropout", type=float, default=0.15)

    parser.add_argument("--lambda_reason", type=float, default=0.001)

    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(

        "--model",

        choices=("geomo", "gcn", "gcn_sdn", "gin", "gin_sdn", "gat", "gat_sdn", "mace"),

        default="geomo",

        help="geomo=EnhancedOG_PGAT; gcn/gin/gat=Pure*; *_sdn=old baselines; mace=MACE",

    )

    parser.add_argument(

        "--prism_ablation",

        type=str,

        default="full",

        choices=("full", "random_scalar", "attention", "depth_embedding", "uniform"),

        help="Table (B) topology mixing; only for --model geomo.",

    )

    parser.add_argument("--n_runs", type=int, default=1, help="Repeat training with seeds seed, seed+1, ...")

    args = parse_args_with_gat_config(parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    dataset = load_freesolv(args.csv, args.limit)

    dataset = attach_positions(dataset)

    input_dim = dataset[0].x.size(-1)

    n_runs = max(1, int(args.n_runs))

    run_rmses, run_maes = [], []

    for run in range(n_runs):

        run_seed = args.seed + run

        set_seed(run_seed)

        if n_runs > 1:

            print(f"\n===== FreeSolv run {run + 1}/{n_runs} | seed={run_seed} =====")

        tr_idx, va_idx, te_idx = scaffold_split(dataset, seed=run_seed)

        tr_ds = [dataset[i] for i in tr_idx]

        va_ds = [dataset[i] for i in va_idx]

        te_ds = [dataset[i] for i in te_idx]

        y_mean, y_std = prepare_label_normalization(tr_ds, [va_ds, te_ds])

        print(f"Label normalization: mean={y_mean:.4f}, std={y_std:.4f}")

        trL = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True)

        vaL = DataLoader(va_ds, batch_size=args.batch)

        teL = DataLoader(te_ds, batch_size=args.batch)

        model = build_model(args, input_dim, 1, device)

        if run == 0:

            print(f"Model: {args.model} | params={sum(p.numel() for p in model.parameters()):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        sched = warmup_cosine_scheduler(opt, warmup=10, total=args.epochs)

        best_mae, best_state, bad = float("inf"), None, 0

        loss_history, val_mae_history = [], []

        for ep in range(1, args.epochs + 1):

            start_time = time.time()

            loss = train_epoch(model, trL, opt, device, args.lambda_reason, 0.0)

            val_mae, val_rmse = evaluate(model, vaL, device, y_mean, y_std, 0.0)

            sched.step()

            end_time = time.time()

            loss_history.append(loss)

            val_mae_history.append(val_mae)

            print(f"[{ep:03d}] Loss={loss:.4f} | Val MAE={val_mae:.4f} | RMSE={val_rmse:.4f} | Time Cost={(end_time-start_time):.2f}s")

            if val_mae < best_mae:

                best_mae = val_mae

                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

                bad = 0

            else:

                bad += 1

            if bad >= args.patience:

                print("Early stopping triggered.")

                break

        if best_state:

            model.load_state_dict(best_state)

        test_mae, test_rmse = evaluate(model, teL, device, y_mean, y_std, 0.0)

        print(f"Final Test MAE={test_mae:.4f} | RMSE={test_rmse:.4f} (best Val MAE={best_mae:.4f})")

        run_rmses.append(float(test_rmse))

        run_maes.append(float(test_mae))

        loss_conv = find_convergence_epoch(loss_history)

        val_conv = find_convergence_epoch(val_mae_history)

        print(f"Convergence: Loss @ epoch {loss_conv if loss_conv else 'N/A'}, "

              f"Val MAE @ epoch {val_conv if val_conv else 'N/A'} "

              f"(total {len(loss_history)} epochs)")

    if n_runs > 1 and run_rmses:

        r = np.asarray(run_rmses, dtype=float)

        m = np.asarray(run_maes, dtype=float)

        r_std = float(r.std(ddof=1)) if len(r) > 1 else 0.0

        m_std = float(m.std(ddof=1)) if len(m) > 1 else 0.0

        print(f"\nFreeSolv Test RMSE over {n_runs} runs: mean={r.mean():.4f}, std={r_std:.4f}, values={run_rmses}")

        print(f"FreeSolv Test MAE over {n_runs} runs: mean={m.mean():.4f}, std={m_std:.4f}, values={run_maes}")

if __name__ == "__main__":

    main()
