

import argparse

from utils.config_utils import parse_args_with_gat_config

import os

import random

import time

import numpy as np

import torch

import torch.nn.functional as F

from torch_geometric.datasets import LRGBDataset

from torch_geometric.loader import DataLoader

from model.sdn import EnhancedOG_PGAT

def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

@torch.no_grad()

def evaluate_mae_and_rf(model, loader, device, max_layers):

    """
    评估 MAE 的同时，计算基于曲率的平均有效感受野 (Mean Receptive Field Depth)
    """

    model.eval()

    ys, ps = [], []

    total_expected_depth = 0.0

    total_nodes = 0

    for batch in loader:

        batch = batch.to(device)

        out = model(batch, return_embedding=True)

        if len(out) == 3:

            pred, kappa, _ = out

        else:

            pred, kappa = out[0], out[1]

        y = batch.y.view(pred.shape)

        ys.append(y.cpu())

        ps.append(pred.cpu())

        layers_idx = torch.arange(max_layers, device=device).float()

        if kappa.dim() == 1:

            kappa = kappa.unsqueeze(-1)

        halt_logits = kappa * layers_idx

        halting_probs = F.softmax(halt_logits, dim=1)

        expected_depth = (halting_probs * layers_idx).sum(dim=1) + 1.0

        total_expected_depth += expected_depth.sum().item()

        total_nodes += kappa.size(0)

    y = torch.cat(ys, dim=0)

    p = torch.cat(ps, dim=0)

    mae = (p - y).abs().mean().item()

    mean_rf = total_expected_depth / max(total_nodes, 1)

    return mae, mean_rf

def main():

    parser = argparse.ArgumentParser(description="Train Spectral-Geometric Model on LRGB peptides-struct.")

    parser.add_argument("--root", type=str, default="data/LRGB", help="Dataset root for LRGBDataset.")

    parser.add_argument("--dataset_name", type=str, default="peptides-struct", help="LRGB dataset name.")

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--layers", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--save_ckpt", type=str, default="checkpoints/peptides_struct_spectral.pt")

    parser.add_argument("--warmup_epochs", type=int, default=10, help="前 N 个 epoch 冻结深度参数")

    parser.add_argument("--depth_lr", type=float, default=1e-5, help="深度参数的专属极小学习率")

    args = parse_args_with_gat_config(parser)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LRGBDataset(root=args.root, name=args.dataset_name, split="train")

    val_ds = LRGBDataset(root=args.root, name=args.dataset_name, split="val")

    test_ds = LRGBDataset(root=args.root, name=args.dataset_name, split="test")

    print(f"Loaded dataset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    sample = train_ds[0]

    in_dim = sample.x.size(-1)

    out_dim = int(sample.y.numel())

    edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else 1

    print(f"Model dims: input_dim={in_dim}, edge_dim={edge_dim}, num_tasks={out_dim}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    model = EnhancedOG_PGAT(

        input_dim=in_dim,

        hidden=args.hidden,

        n_layers=args.layers,

        dropout=args.dropout,

        num_tasks=out_dim,

        edge_dim=edge_dim

    ).to(device)

    depth_param_keywords = ['kappa', 'base_depth', 'halt']

    depth_params = []

    other_params = []

    for name, param in model.named_parameters():

        if any(keyword in name for keyword in depth_param_keywords):

            depth_params.append(param)

            print(f"Found depth parameter: {name} (will use lr={args.depth_lr})")

        else:

            other_params.append(param)

    optimizer = torch.optim.AdamW([

        {'params': other_params, 'lr': args.lr, 'weight_decay': args.weight_decay},

        {'params': depth_params, 'lr': args.depth_lr, 'weight_decay': 0.0}

    ])

    best_val_mae = float("inf")

    best_test_mae = float("inf")

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)

    epoch_times = []

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        if epoch <= args.warmup_epochs:

            for p in depth_params:

                p.requires_grad = False

            if epoch == 1:

                print(f"--- Epoch 1 to {args.warmup_epochs}: Depth parameters FROZEN for warm-up ---")

        elif epoch == args.warmup_epochs + 1:

            for p in depth_params:

                p.requires_grad = True

            print(f"--- Epoch {epoch}: Depth parameters UNFROZEN. Self-adaptive depth starts! ---")

        model.train()

        total_loss = 0.0

        total_graphs = 0

        for batch in train_loader:

            batch = batch.to(device)

            out = model(batch)

            pred = out[0] if isinstance(out, tuple) else out

            y = batch.y.view(pred.shape)

            task_loss = F.l1_loss(pred, y)

            phys_reg_loss = 0.0

            if hasattr(model, 'get_curvature_reg'):

                phys_reg_loss = model.get_curvature_reg()

            loss = task_loss + phys_reg_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += float(task_loss.item()) * batch.num_graphs

            total_graphs += batch.num_graphs

        train_mae = total_loss / max(total_graphs, 1)

        val_mae, val_rf = evaluate_mae_and_rf(model, val_loader, device, max_layers=max(args.layers, 5))

        test_mae, test_rf = evaluate_mae_and_rf(model, test_loader, device, max_layers=max(args.layers, 5))

        if val_mae < best_val_mae:

            best_val_mae = val_mae

            best_test_mae = test_mae

            torch.save({

                "model": model.state_dict(),

                "args": vars(args),

                "best_val_mae": best_val_mae,

                "test_mae_at_best_val": best_test_mae,

            }, args.save_ckpt)

        epoch_sec = time.time() - start_time

        epoch_times.append(epoch_sec)

        avg_epoch_sec = sum(epoch_times) / len(epoch_times)

        print(

            f"[{epoch:03d}] Train MAE: {train_mae:.4f} | "

            f"Val MAE: {val_mae:.4f} (RF Depth: {val_rf:.2f}) | "

            f"Test MAE: {test_mae:.4f} (RF Depth: {test_rf:.2f}) | "

            f"Best Val: {best_val_mae:.4f} | "

            f"Epoch Time: {epoch_sec:.2f}s | Avg Epoch Time: {avg_epoch_sec:.2f}s"

        )

    if os.path.exists(args.save_ckpt):

        checkpoint = torch.load(args.save_ckpt)

        model.load_state_dict(checkpoint["model"])

    final_test_mae, final_test_rf = evaluate_mae_and_rf(model, test_loader, device, max_layers=max(args.layers, 5))

    print("\n" + "="*50)

    print("TRAINING COMPLETED")

    print("="*50)

    print(f"Final Test MAE = {final_test_mae:.4f} | Best Val MAE = {best_val_mae:.4f}")

    print(f"Mean Receptive Field (Test Dataset): {final_test_rf:.2f}")

    if len(epoch_times) > 0:

        total_train_time = sum(epoch_times)

        print(f"Total Train Time: {total_train_time/60:.2f} min | Avg/Epoch: {np.mean(epoch_times):.2f} s")

    print("="*50)

if __name__ == "__main__":

    main()
