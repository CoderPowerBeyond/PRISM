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

def load_pretrained_backbone_by_shape(model, ckpt_path):

    """
    Load checkpoint keys that match by both name and tensor shape.
    Useful for transfer when output heads/task dims differ.
    """

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    if not isinstance(state, dict):

        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")

    model_state = model.state_dict()

    loadable = {}

    loaded, skipped = 0, 0

    for k, v in state.items():

        if k in model_state and model_state[k].shape == v.shape:

            loadable[k] = v

            loaded += 1

        else:

            skipped += 1

    model_state.update(loadable)

    model.load_state_dict(model_state, strict=False)

    return loaded, skipped

def attach_z_for_mace(data: Data) -> None:

    """Integer atomic numbers for MACE. PyG-style: x[:,0] often encodes Z (or index == Z)."""

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

    crit = nn.BCEWithLogitsLoss(reduction="none")

    total, count = 0.0, 0

    for batch in loader:

        batch = batch.to(device)

        if noise_level > 0:

            batch.x = batch.x + noise_level * torch.randn_like(batch.x)

        optimizer.zero_grad(set_to_none=True)

        out, kappa = model(batch)

        y = batch.y.float()

        mask = (y != -1) & ~torch.isnan(y)

        if mask.sum() == 0:

            continue

        out = out.float().clamp(-15, 15)

        if pos_weight is not None:

            pw = torch.nan_to_num(pos_weight.clone().detach(), 1.0)

        else:

            with torch.no_grad():

                n_tasks = y.shape[1]

                w = []

                for i in range(n_tasks):

                    yy, m = y[:, i], mask[:, i]

                    if m.sum() == 0:

                        w.append(torch.tensor(1.0, device=device))

                        continue

                    pos, neg = (yy[m] > 0.5).sum(), (yy[m] <= 0.5).sum()

                    w.append((neg + 1) / (pos + 1))

                pw = torch.stack(w)

        loss_mat = crit(out, y.nan_to_num(0.0))

        loss_mat = loss_mat * torch.where(y > 0.5, pw, 1.0)

        main = loss_mat[mask].mean()

        curv_pen = laplace_beltrami_penalty(kappa, batch.edge_index)

        loss = main + lam * (model.get_curvature_reg() + curv_pen)

        if not torch.isfinite(loss):

            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()

        total += float(loss.detach().cpu())

        count += 1

    return total / max(count, 1)

@torch.no_grad()

def eval_epoch(model, loader, device, num_tasks):

    model.eval()

    y_true, y_pred = [], []

    for b in loader:

        b = b.to(device)

        out, _ = model(b)

        y = b.y.float()

        mask = (y != -1)

        if mask.sum() == 0:

            continue

        y_true.append(y[mask].cpu())

        y_pred.append(torch.sigmoid(out[mask].float().clamp(-20, 20)).cpu())

    if not y_true:

        return 0.0

    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()

    aucs = []

    if y_true.ndim == 1 or y_true.shape[1] == 1:

        try:

            return roc_auc_score(y_true, y_pred)

        except Exception:

            return 0.0

    for i in range(num_tasks):

        yi, pi = y_true[:, i], y_pred[:, i]

        if len(np.unique(yi)) < 2:

            continue

        try:

            aucs.append(roc_auc_score(yi, pi))

        except Exception:

            pass

    return float(np.nanmean(aucs)) if aucs else 0.0

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", default="clintox")

    ap.add_argument(

        "--pretrain_ckpt",

        type=str,

        default="",

        help="Optional QM9 pretrained SDN checkpoint path. Loads matching backbone weights.",

    )

    ap.add_argument("--epochs", type=int, default=500)

    ap.add_argument("--batch", type=int, default=64)

    ap.add_argument("--lr", type=float, default=5e-4)

    ap.add_argument("--hidden", type=int, default=256)

    ap.add_argument("--layers", type=int, default=5)

    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--lambda_reason", type=float, default=0.00015)

    ap.add_argument("--patience", type=int, default=60)

    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument(

        "--model",

        choices=("geomo", "mace"),

        default="geomo",

        help="geomo=EnhancedOG_PGAT, mace=MACE baseline (requires pip install mace-torch)",

    )

    args = parse_args_with_gat_config(ap)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pth_path = os.path.join("processed", f"{args.dataset}.pth")

    if not os.path.exists(pth_path):

        pth_path = os.path.join("data/processed", f"{args.dataset}.pth")

    print(f"Loading DGL dataset from {pth_path}")

    raw = torch.load(pth_path, map_location="cpu")

    required = ["train_graph_list", "valid_graph_list", "test_graph_list",

                "train_label", "valid_label", "test_label"]

    if not all(k in raw for k in required):

        raise ValueError(f"Unexpected dataset structure: {list(raw.keys())}")

    print("Detected DGL-style dataset structure.")

    def dgl_to_pyg(glist, ylist):

        data_list = []

        for g, y in zip(glist, ylist):

            if not isinstance(g, dgl.DGLGraph):

                raise TypeError(f"Expected DGLGraph, got {type(g)}")

            if 'feat' in g.ndata:

                x = g.ndata['feat'].float()

            elif 'h' in g.ndata:

                x = g.ndata['h'].float()

            else:

                raise KeyError("Missing node features")

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

            data_list.append(Data(x=x, edge_index=edge_index, y=y_tensor, pos=pos))

        return data_list

    tr_ds = dgl_to_pyg(raw["train_graph_list"], raw["train_label"])

    va_ds = dgl_to_pyg(raw["valid_graph_list"], raw["valid_label"])

    te_ds = dgl_to_pyg(raw["test_graph_list"], raw["test_label"])

    print(f"Converted DGL->PyG successfully: train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")

    num_feat = tr_ds[0].x.shape[1]

    num_tasks = tr_ds[0].y.shape[-1]

    trL = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)

    vaL = DataLoader(va_ds, batch_size=args.batch)

    teL = DataLoader(te_ds, batch_size=args.batch)

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

        ).to(device)

    if args.pretrain_ckpt:

        if args.model == "mace":

            print("--pretrain_ckpt is ignored for --model mace (different architecture).")

        else:

            if not os.path.isfile(args.pretrain_ckpt):

                raise FileNotFoundError(f"--pretrain_ckpt not found: {args.pretrain_ckpt}")

            loaded, skipped = load_pretrained_backbone_by_shape(model, args.pretrain_ckpt)

            print(f"Loaded pretrained weights from {args.pretrain_ckpt}")

            print(f"   matched keys: {loaded}, skipped keys: {skipped}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr / 20)

    all_y = torch.cat([d.y.view(-1) for d in tr_ds + va_ds + te_ds])

    valid_y = all_y[all_y != -1]

    pos_w = None

    if (valid_y == 1).sum() > 0:

        y0, y1 = (valid_y == 0).sum().float(), (valid_y == 1).sum().float()

        pos_w = torch.tensor([y0 / (y1 + 1e-6)], device=device)

        print(f"→ pos_weight = {pos_w.item():.3f}")

    best_auc, best_state, bad = 0.0, None, 0

    for ep in range(1, args.epochs + 1):

        t0 = time.perf_counter()

        loss_tr = train_epoch(model, trL, opt, device, args.lambda_reason, pos_w)

        auc_val = eval_epoch(model, vaL, device, num_tasks)

        sched.step()

        dt = time.perf_counter() - t0

        print(f"[{ep:03d}] Loss={loss_tr:.4f} | Val AUC={auc_val:.4f} | Time {dt:.2f}s")

        if auc_val > best_auc:

            best_auc, best_state, bad = auc_val, model.state_dict(), 0

        else:

            bad += 1

        if bad >= args.patience:

            print("Early stopping.")

            break

    if best_state:

        model.load_state_dict(best_state)

    auc_test = eval_epoch(model, teL, device, num_tasks)

    print(f"Final Test AUC={auc_test:.4f} (best Val AUC={best_auc:.4f})")

if __name__ == "__main__":

    main()
