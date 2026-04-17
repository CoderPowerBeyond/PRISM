

import os

import argparse

from utils.config_utils import parse_args_with_gat_config

import random

import time

import numpy as np

from sklearn.metrics import roc_auc_score

import torch

import torch.nn as nn

from torch_geometric.loader import DataLoader

from torch_geometric.data import Data

import dgl

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

def attach_z_for_mace(data: Data) -> None:

    """Integer atomic numbers for MACE. PyG from_smiles: x[:,0] indexes atomic_num (0..118) == Z."""

    if getattr(data, "z", None) is not None:

        return

    x = data.x

    if x.dim() != 2 or x.size(0) == 0:

        return

    c0 = x[:, 0]

    if c0.dtype.is_floating_point:

        c0 = c0.round()

    data.z = c0.long().clamp(1, 118)

def attach_z_for_mace_list(data_list: list) -> None:

    for d in data_list:

        attach_z_for_mace(d)

def reasoning_loss(kappa, edge_index, lam=0.001, device="cpu"):

    if kappa is None:

        return torch.tensor(0.0, device=device)

    return lam * laplace_beltrami_penalty(kappa, edge_index)

def train_epoch(model, loader, optimizer, device, lam, pos_weight=None, noise_level=0.0):

    model.train()

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    total_loss, count = 0.0, 0

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn(batch.x.shape, device=batch.x.device)

        optimizer.zero_grad(set_to_none=True)

        out, kappa = model(batch)

        out = torch.clamp(out.float(), -15.0, 15.0)

        y = batch.y.float()

        mask = (y != -1) & ~torch.isnan(y)

        if mask.sum() == 0:

            continue

        if pos_weight is not None:

            pw = pos_weight.clone().detach()

            pw = torch.nan_to_num(pw, nan=1.0, posinf=1.0, neginf=1.0)

        else:

            with torch.no_grad():

                n_tasks = y.shape[1]

                pw_list = []

                for i in range(n_tasks):

                    yi = y[:, i]

                    valid = mask[:, i]

                    if valid.sum() == 0:

                        pw_list.append(torch.tensor(1.0, device=device))

                        continue

                    pos = (yi[valid] > 0.5).sum().float()

                    neg = (yi[valid] <= 0.5).sum().float()

                    pw_list.append((neg + 1) / (pos + 1))

                pw = torch.stack(pw_list)

        loss_mat = criterion(out, y.nan_to_num(0.0))

        loss_mat = loss_mat * torch.where(y > 0.5, pw, 1.0)

        loss_main = loss_mat[mask].mean()

        curv_pen = laplace_beltrami_penalty(kappa, getattr(batch, "edge_index", None))

        curv_pen = curv_pen if torch.isfinite(curv_pen) else torch.tensor(0.0, device=device)

        loss = loss_main + lam * (model.get_curvature_reg() + curv_pen)

        if not torch.isfinite(loss):

            print("NaN detected, skipping batch.")

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        for p in model.parameters():

            if torch.isnan(p).any() or torch.isinf(p).any():

                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

        total_loss += float(loss.detach().cpu())

        count += 1

    return total_loss / max(count, 1)

@torch.no_grad()

def eval_epoch(model, loader, device, num_tasks, noise_level=0.0):

    model.eval()

    y_true, y_pred = [], []

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x.float() + noise_level * torch.randn(batch.x.shape, device=batch.x.device)

        out, _ = model(batch)

        out = out.float().clamp(-20, 20)

        y = batch.y.float()

        mask = (y != -1)

        if mask.sum() == 0:

            continue

        y_true.append(y[mask].cpu())

        y_pred.append(torch.sigmoid(out[mask]).cpu())

    if not y_true:

        return 0.0

    y_true, y_pred = torch.cat(y_true, 0).numpy(), torch.cat(y_pred, 0).numpy()

    def safe_auc(y, p):

        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

    aucs = []

    if y_true.ndim == 1 or y_true.shape[1] == 1:

        return float(np.nanmax([safe_auc(y_true, y_pred), safe_auc(y_true, 1 - y_pred)]))

    for i in range(num_tasks):

        yi, pi = y_true[:, i], y_pred[:, i]

        if len(np.unique(yi)) < 2:

            continue

        aucs.append(np.nanmax([safe_auc(yi, pi), safe_auc(yi, 1 - pi)]))

    return float(np.nanmean(aucs)) if aucs else 0.0

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="sider",

                    choices=["sider", "bbbp", "bace", "tox21", "clintox", "hiv"])

    ap.add_argument("--epochs", type=int, default=600)

    ap.add_argument("--batch", type=int, default=128)

    ap.add_argument("--lr", type=float, default=0.00001)

    ap.add_argument("--hidden", type=int, default=256)

    ap.add_argument("--layers", type=int, default=6)

    ap.add_argument("--dropout", type=float, default=0.05)

    ap.add_argument("--lambda_reason", type=float, default=0.00015)

    ap.add_argument("--patience", type=int, default=60)

    ap.add_argument("--seed", type=int, default=7)

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

        help="Topological diffusion / readout ablation; only used when --model geomo.",

    )

    ap.add_argument(

        "--n_runs",

        type=int,

        default=1,

        help="Repeat training with seeds seed, seed+1, ... (same fixed split from .pth).",

    )

    args = parse_args_with_gat_config(ap)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    pth_path = os.path.join("processed", f"{args.dataset}.pth")

    if not os.path.exists(pth_path):

        pth_path = os.path.join("data/processed", f"{args.dataset}.pth")

    print(f"Loading DGL dataset from {pth_path}")

    raw = torch.load(pth_path, map_location="cpu")

    required_keys = ["train_graph_list", "valid_graph_list", "test_graph_list",

                     "train_label", "valid_label", "test_label"]

    if all(k in raw for k in required_keys):

        print("Detected DGL-style dataset structure.")

        def dgl_to_pyg(glist, ylist):

            """
            Convert list of DGLGraph + labels to PyG Data objects.
            Collects node coordinates from 'coor' -> data.pos
            and node features from 'feat' -> data.x
            """

            data_list = []

            for g, y in zip(glist, ylist):

                if not isinstance(g, dgl.DGLGraph):

                    raise TypeError(f"Expected DGLGraph, got {type(g)}")

                if 'feat' in g.ndata:

                    x = g.ndata['feat'].float()

                elif 'h' in g.ndata:

                    x = g.ndata['h'].float()

                else:

                    raise KeyError("Missing node features: expected g.ndata['feat'] or g.ndata['h']")

                if 'coor' in g.ndata:

                    pos = g.ndata['coor'].float()

                elif 'coord' in g.ndata:

                    pos = g.ndata['coord'].float()

                elif 'pos' in g.ndata:

                    pos = g.ndata['pos'].float()

                else:

                    pos = torch.zeros((g.num_nodes(), 3), dtype=torch.float)

                src, dst = g.edges()

                edge_index = torch.stack([src, dst], dim=0).long()

                y_tensor = y.view(1, -1).float()

                data = Data(x=x, edge_index=edge_index, y=y_tensor, pos=pos)

                data_list.append(data)

            return data_list

        tr_ds = dgl_to_pyg(raw["train_graph_list"], raw["train_label"])

        va_ds = dgl_to_pyg(raw["valid_graph_list"], raw["valid_label"])

        te_ds = dgl_to_pyg(raw["test_graph_list"], raw["test_label"])

        print(f"Converted DGL->PyG successfully: train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")

    else:

        raise ValueError(f"Unexpected .pth structure: {list(raw.keys())}")

    if args.model == "mace":

        attach_z_for_mace_list(tr_ds)

        attach_z_for_mace_list(va_ds)

        attach_z_for_mace_list(te_ds)

    dataset = tr_ds + va_ds + te_ds

    num_feat = dataset[0].x.shape[1]

    num_tasks = dataset[0].y.shape[-1]

    print(f"Feature dim = {num_feat}, Tasks = {num_tasks}")

    trL = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)

    vaL = DataLoader(va_ds, batch_size=args.batch)

    teL = DataLoader(te_ds, batch_size=args.batch)

    all_y = torch.cat([d.y.view(-1) for d in dataset], 0)

    valid_y = all_y[all_y != -1]

    pos_w = None

    if (valid_y == 1).sum() > 0:

        y0, y1 = (valid_y == 0).sum().float(), (valid_y == 1).sum().float()

        pos_w = torch.tensor([y0 / (y1 + 1e-6)], device=device)

        print(f"→ pos_weight = {pos_w.item():.3f}")

    n_runs = max(1, int(args.n_runs))

    run_aucs: list[float] = []

    for run in range(n_runs):

        run_seed = args.seed + run

        set_seed(run_seed)

        if n_runs > 1:

            print(f"\n===== Run {run + 1}/{n_runs} | seed={run_seed} =====")

        if args.model == "mace":

            if build_mace_for_geomo is None:

                raise RuntimeError("MACE requires: pip install mace-torch")

            model = build_mace_for_geomo(

                input_dim=num_feat,

                hidden=args.hidden,

                n_layers=args.layers,

                dropout=args.dropout,

                num_tasks=num_tasks,

            ).to(device)

        else:

            model = EnhancedOG_PGAT(

                input_dim=num_feat,

                hidden=args.hidden,

                n_layers=args.layers,

                dropout=args.dropout,

                num_tasks=num_tasks,

                disable_crf=False,

                mean_pool_only=False,

                use_se3=True,

                prism_ablation=args.prism_ablation,

            ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)

        best_auc, best_state, bad = 0.0, None, 0

        for ep in range(1, args.epochs + 1):

            t0 = time.perf_counter()

            tr_loss = train_epoch(model, trL, opt, device, args.lambda_reason, pos_w, 0.0)

            val_auc = eval_epoch(model, vaL, device, num_tasks, 0.0)

            sched.step()

            dt = time.perf_counter() - t0

            print(f"[{ep:03d}] Loss={tr_loss:.4f} | Val AUC={val_auc:.4f} | Time {dt:.2f}s")

            if val_auc > best_auc:

                best_auc, best_state, bad = val_auc, model.state_dict(), 0

            else:

                bad += 1

            if bad >= args.patience:

                print("Early stopping (no improvement).")

                break

        if best_state:

            model.load_state_dict(best_state)

        test_auc = eval_epoch(model, teL, device, num_tasks, 0.0)

        print(f"Final Test AUC = {test_auc:.4f} (best Val AUC = {best_auc:.4f})")

        run_aucs.append(float(test_auc))

    if n_runs > 1 and run_aucs:

        a = np.asarray(run_aucs, dtype=float)

        a_std = float(a.std(ddof=1)) if len(a) > 1 else 0.0

        print(

            f"\nTest AUC over {n_runs} runs: mean={a.mean():.4f}, std={a_std:.4f}, values={run_aucs}"

        )

if __name__ == "__main__":

    main()
