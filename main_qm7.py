

"""
main_qm7_sdn.py
================

Train and evaluate the EnhancedOG_PGAT (SDN) model on QM7 data.
Includes RDKit 3D coordinate generation, RBF expansion for 3D edges,
Test set evaluation, and Visualization utilities.
"""

import os

import torch

import torch.nn.functional as F

from torch_geometric.loader import DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import argparse

from utils.config_utils import parse_args_with_gat_config

try:

    from rdkit import Chem

    from rdkit.Chem import AllChem

except ImportError:

    print("RDKit is not installed. Please install it via: conda install -c conda-forge rdkit")

from qm7_dataset import QM7Dataset

from model.sdn import EnhancedOG_PGAT, laplace_beltrami_penalty

try:

    from model.mace import build_mace_for_geomo

except ImportError:

    build_mace_for_geomo = None

def rbf_expansion(distances, num_centers=16, d_min=0.0, d_max=10.0, device='cpu'):

    """Expand 1D distances to 16D using Radial Basis Functions (RBF)"""

    centers = torch.linspace(d_min, d_max, num_centers, device=device)

    gamma = 1.0 / ((d_max - d_min) / num_centers) ** 2

    return torch.exp(-gamma * (distances - centers) ** 2)

def ensure_edge_attr(data, device):

    """Ensures edge_attr_2d (3 dims) and edge_attr_3d (16 dims) exist"""

    num_edges = data.edge_index.size(1) if data.edge_index is not None else 0

    if getattr(data, 'edge_attr_2d', None) is None:

        if getattr(data, 'edge_attr', None) is not None:

            attr = data.edge_attr.clone().float()

            if attr.dim() == 1:

                attr = attr.view(-1, 1)

        else:

            attr = torch.ones((num_edges, 1), dtype=torch.float32, device=device)

        if attr.size(1) < 3:

            padding = torch.zeros((num_edges, 3 - attr.size(1)), device=device)

            data.edge_attr_2d = torch.cat([attr, padding], dim=1)

        elif attr.size(1) > 3:

            data.edge_attr_2d = attr[:, :3]

        else:

            data.edge_attr_2d = attr

    else:

        if data.edge_attr_2d.dim() == 1:

            data.edge_attr_2d = data.edge_attr_2d.view(-1, 1)

        if data.edge_attr_2d.size(1) < 3:

            padding = torch.zeros((num_edges, 3 - data.edge_attr_2d.size(1)), device=device)

            data.edge_attr_2d = torch.cat([data.edge_attr_2d, padding], dim=1)

        elif data.edge_attr_2d.size(1) > 3:

            data.edge_attr_2d = data.edge_attr_2d[:, :3]

    if getattr(data, 'edge_attr_3d', None) is None:

        if getattr(data, 'pos', None) is not None and data.edge_index is not None:

            row, col = data.edge_index

            dist = torch.norm(data.pos[row] - data.pos[col] + 1e-8, p=2, dim=-1, keepdim=True)

            data.edge_attr_3d = rbf_expansion(dist, num_centers=16, device=device)

        else:

            data.edge_attr_3d = torch.zeros((num_edges, 16), dtype=torch.float32, device=device)

    return data

def add_3d_coords_to_dataset(dataset):

    """Iterates through the dataset and adds 3D positions using RDKit if missing"""

    print("Checking/Generating 3D coordinates using RDKit...")

    missing_smiles_count = 0

    for i in range(len(dataset)):

        data = dataset[i]

        if getattr(data, 'pos', None) is None:

            if hasattr(data, 'smiles') and data.smiles:

                smiles = data.smiles if isinstance(data.smiles, str) else data.smiles[0]

                mol = Chem.MolFromSmiles(smiles)

                if mol is not None:

                    mol = Chem.AddHs(mol)

                    AllChem.EmbedMolecule(mol, randomSeed=42)

                    try:

                        AllChem.MMFFOptimizeMolecule(mol)

                    except:

                        pass

                    conf = mol.GetConformer()

                    pos = [list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())]

                    data.pos = torch.tensor(pos, dtype=torch.float)

                else:

                    data.pos = torch.zeros((data.num_nodes, 3), dtype=torch.float)

            else:

                missing_smiles_count += 1

                data.pos = torch.zeros((data.num_nodes, 3), dtype=torch.float)

    if missing_smiles_count > 0:

        print(f"Warning: {missing_smiles_count} molecules lacked SMILES. Filled with zero coordinates.")

    print("3D coordinates ready.")

def attach_z_for_mace(dataset):

    """Attach atomic numbers tensor (z) required by MACE."""

    missing = 0

    for data in dataset:

        if getattr(data, "z", None) is not None:

            continue

        z = None

        smi = getattr(data, "smiles", None)

        if smi:

            try:

                mol = Chem.MolFromSmiles(smi if isinstance(smi, str) else smi[0])

                if mol is not None:

                    z = torch.tensor(

                        [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())],

                        dtype=torch.long,

                    )

            except Exception:

                z = None

        if z is None:

            x0 = data.x[:, 0]

            if x0.dtype.is_floating_point:

                x0 = x0.round()

            z = x0.long().clamp(1, 118)

        data.z = z

        if data.z.numel() != data.num_nodes:

            missing += 1

    if missing > 0:

        print(f"Warning: {missing} samples had z/node mismatch after fallback.")

def train_one_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0.0

    for data in loader:

        data = data.to(device)

        data = ensure_edge_attr(data, device)

        out, kappa = model(data)

        loss = F.l1_loss(out.squeeze(), data.y.squeeze())

        loss = loss + 1e-4 * (model.get_curvature_reg() + laplace_beltrami_penalty(kappa, getattr(data, "edge_index", None)))

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

@torch.no_grad()

def evaluate(model, loader, device):

    model.eval()

    y_true, y_pred = [], []

    for data in loader:

        data = data.to(device)

        data = ensure_edge_attr(data, device)

        out, _ = model(data)

        y_true.append(data.y.squeeze().cpu())

        y_pred.append(out.squeeze().cpu())

    y_true = torch.cat(y_true).numpy()

    y_pred = torch.cat(y_pred).numpy()

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2, y_true, y_pred

def plot_parity(y_true, y_pred):

    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, y_pred, s=20, alpha=0.7, edgecolors="k")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]

    plt.plot(lims, lims, "r--")

    plt.xlabel("True Energy")

    plt.ylabel("Predicted Energy")

    plt.title("Parity Plot – QM7 SDN")

    plt.tight_layout()

    plt.savefig("partity.png", dpi=300)

    plt.close()

    print("Saved Parity Plot to partity.png")

def plot_curvature_hist(kappa_values):

    vals = torch.cat(kappa_values).detach().cpu().numpy()

    plt.figure(figsize=(6, 4))

    sns.histplot(vals, bins=40, color="#2c7fb8", edgecolor=None)

    plt.xlabel("Node curvature κ")

    plt.ylabel("Frequency")

    plt.title("Distribution of Curvature (GeoReason Field)")

    plt.tight_layout()

    plt.savefig("dis.png", dpi=300)

    plt.close()

def plot_curvature_profile(model, save_path="./", show=False):

    if not hasattr(model, "curv_trace_mean") or not hasattr(model, "curv_trace_var"):

        print("No curvature trace recorded in model — skip.")

        return

    layers = list(range(1, len(model.curv_trace_mean) + 1))

    means = [float(m) for m in model.curv_trace_mean]

    vars_ = [float(v) for v in model.curv_trace_var]

    plt.figure(figsize=(6, 4))

    plt.plot(layers, means, marker="o", color="#1f77b4", label="Mean κ")

    plt.fill_between(

        layers,

        [m - v ** 0.5 for m, v in zip(means, vars_)],

        [m + v ** 0.5 for m, v in zip(means, vars_)],

        color="#1f77b4",

        alpha=0.2,

        label="±√Var"

    )

    plt.xlabel("Layer")

    plt.ylabel("Curvature κ")

    plt.title("Layer‑wise Curvature Profile (GeoReason Field++)")

    plt.grid(alpha=0.3, linestyle="--")

    plt.legend(frameon=False)

    plt.tight_layout()

    plt.savefig("cur_trace.png", dpi=300)

    if show:

        plt.show()

    plt.close()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=400)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--layers", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--model", choices=("geomo", "mace"), default="geomo")

    parser.add_argument(

        "--prism_ablation",

        type=str,

        default="full",

        choices=("full", "random_scalar", "attention", "depth_embedding", "uniform"),

        help="Table (B) topology mixing; only for --model geomo.",

    )

    args = parse_args_with_gat_config(parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = QM7Dataset(root="data/QM7")

    add_3d_coords_to_dataset(dataset)

    if args.model == "mace":

        attach_z_for_mace(dataset)

    y_all = torch.cat([d.y for d in dataset])

    mean_y, std_y = y_all.mean(), y_all.std()

    for d in dataset:

        d.y = (d.y - mean_y) / std_y

    N = len(dataset)

    split1 = int(0.8 * N)

    split2 = int(0.9 * N)

    train_dataset = dataset[:split1]

    val_dataset = dataset[split1:split2]

    test_dataset = dataset[split2:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    input_dim = dataset.num_features

    if args.model == "mace":

        if build_mace_for_geomo is None:

            raise RuntimeError("MACE requires: pip install mace-torch")

        model = build_mace_for_geomo(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=1,

        ).to(device)

    else:

        model = EnhancedOG_PGAT(

            input_dim=input_dim,

            hidden=args.hidden,

            n_layers=args.layers,

            dropout=args.dropout,

            num_tasks=1,

            disable_crf=False,

            mean_pool_only=False,

            use_equivariant=True,

            use_se3=True,

            prism_ablation=args.prism_ablation,

        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_rmse, val_mae, val_r2, _, _ = evaluate(model, val_loader, device)

        print(f"[{epoch}] Loss={train_loss:.4f} | Val RMSE={val_rmse:.4f} | MAE={val_mae:.4f} | R2={val_r2:.4f}")

        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            torch.save(model.state_dict(), "best_qm7_sdn.pt")

    model.load_state_dict(torch.load("best_qm7_sdn.pt"))

    test_rmse, test_mae, test_r2, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"\nFinal Test RMSE = {test_rmse:.4f} | MAE = {test_mae:.4f} | R2 = {test_r2:.4f} | Best Val RMSE = {best_val_rmse:.4f}\n")

    plot_parity(y_true, y_pred)

    kappa_vals = []

    with torch.no_grad():

        for data in test_loader:

            data = data.to(device)

            data = ensure_edge_attr(data, device)

            _, kappa = model(data)

            kappa_vals.append(kappa)

    plot_curvature_hist(kappa_vals)

    plot_curvature_profile(model, save_path="./", show=False)

if __name__ == "__main__":

    main()
