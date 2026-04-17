

import argparse

from utils.config_utils import parse_args_with_gat_config

import os, math, time, random

import torch

import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader

import pandas as pd

import numpy as np

from tqdm import tqdm

from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR

from model.sdn import EnhancedOG_PGAT

from rdkit import Chem

from rdkit.Chem import rdchem

from rdkit.Chem import AllChem

from rdkit.Chem import Descriptors3D

torch.manual_seed(42)

np.random.seed(42)

random.seed(42)

ATOM_LIST = list(range(1, 119))

BOND_TYPES = [rdchem.BondType.SINGLE,

              rdchem.BondType.DOUBLE,

              rdchem.BondType.TRIPLE,

              rdchem.BondType.AROMATIC]

def atom_features(atom):

    """Comprehensive atom feature vector (~11 dims)."""

    return torch.tensor([

        atom.GetAtomicNum(),

        atom.GetTotalDegree(),

        atom.GetFormalCharge(),

        atom.GetTotalNumHs(),

        atom.GetHybridization().real,

        float(atom.GetIsAromatic()),

        atom.GetMass(),

        atom.GetTotalValence(),

        atom.GetImplicitValence(),

        atom.GetNumRadicalElectrons(),

        atom.GetChiralTag(),

    ], dtype=torch.float)

def bond_features(bond):

    """Bond one-hot + stereo + in_ring (6 dims)."""

    bt = [float(bond.GetBondTypeAsDouble() == b) for b in [1, 2, 3, 1.5]]

    stereo = bond.GetStereo()

    in_ring = float(bond.IsInRing())

    return torch.tensor(bt + [stereo, in_ring], dtype=torch.float)

def smiles_to_graph_3d(smiles):

    """Parse SMILES → PyG Data with 3D geometry."""

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:

        return None

    mol = Chem.AddHs(mol)

    try:

        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        AllChem.UFFOptimizeMolecule(mol)

    except Exception:

        return None

    conf = mol.GetConformer()

    num_atoms = mol.GetNumAtoms()

    positions = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(num_atoms)],

                             dtype=torch.float)

    x = torch.stack([atom_features(a) for a in mol.GetAtoms()])

    row, col, edge_attr = [], [], []

    for bond in mol.GetBonds():

        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        b_feat = bond_features(bond)

        dist = torch.norm(positions[i] - positions[j])

        full_feat = torch.cat([b_feat, dist.reshape(1)], dim=0)

        row += [i, j]

        col += [j, i]

        edge_attr.append(full_feat)

        edge_attr.append(full_feat)

    if not row:

        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)

    edge_attr = torch.stack(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=positions)

    return data

def rbf_expansion(distances, num_centers=16, d_min=0.0, d_max=10.0, device='cpu'):

    """Expand 1D distances to 16D using Radial Basis Functions (RBF)"""

    centers = torch.linspace(d_min, d_max, num_centers, device=device)

    gamma = 1.0 / ((d_max - d_min) / num_centers) ** 2

    return torch.exp(-gamma * (distances.unsqueeze(-1) - centers) ** 2)

def ensure_edge_attr(data, device):

    """Ensures edge_attr_2d (7 dims) and edge_attr_3d (16 dims) exist to match model expectations"""

    num_edges = data.edge_index.size(1) if data.edge_index is not None else 0

    if getattr(data, 'edge_attr_2d', None) is None:

        if getattr(data, 'edge_attr', None) is not None:

            attr = data.edge_attr.clone().float()

            if attr.dim() == 1:

                attr = attr.view(-1, 1)

        else:

            attr = torch.ones((num_edges, 1), dtype=torch.float32, device=device)

        if attr.size(1) < 7:

            padding = torch.zeros((num_edges, 7 - attr.size(1)), device=device)

            data.edge_attr_2d = torch.cat([attr, padding], dim=1)

        elif attr.size(1) > 7:

            data.edge_attr_2d = attr[:, :7]

        else:

            data.edge_attr_2d = attr

    else:

        if data.edge_attr_2d.dim() == 1:

            data.edge_attr_2d = data.edge_attr_2d.view(-1, 1)

        if data.edge_attr_2d.size(1) < 7:

            padding = torch.zeros((num_edges, 7 - data.edge_attr_2d.size(1)), device=device)

            data.edge_attr_2d = torch.cat([data.edge_attr_2d, padding], dim=1)

        elif data.edge_attr_2d.size(1) > 7:

            data.edge_attr_2d = data.edge_attr_2d[:, :7]

    if getattr(data, 'edge_attr_3d', None) is None:

        if getattr(data, 'pos', None) is not None and data.edge_index is not None:

            row, col = data.edge_index

            dist = torch.norm(data.pos[row] - data.pos[col] + 1e-8, p=2, dim=-1)

            data.edge_attr_3d = rbf_expansion(dist, num_centers=16, device=device)

        else:

            data.edge_attr_3d = torch.zeros((num_edges, 16), dtype=torch.float32, device=device)

    return data

def load_qm9_csv_highfidelity(path="data/qm9.csv", target="Gap", limit=None):

    df = pd.read_csv(path)

    if limit:

        df = df.iloc[:limit]

    dataset = []

    for smi, val in tqdm(zip(df["SMILES"], df[target]), total=len(df)):

        g = smiles_to_graph_3d(smi)

        if g is not None:

            g.y = torch.tensor([float(val)], dtype=torch.float)

            dataset.append(g)

    return dataset

def train_epoch(model, loader, opt, device):

    model.train()

    total = 0

    for batch in loader:

        batch = batch.to(device)

        batch = ensure_edge_attr(batch, device)

        pred, _ = model(batch)

        loss = F.l1_loss(pred.view(-1), batch.y.view(-1))

        reg = model.get_curvature_reg()

        loss = loss + 1e-4 * reg

        opt.zero_grad()

        loss.backward()

        opt.step()

        total += loss.item() * batch.num_graphs

    return total / len(loader.dataset)

@torch.no_grad()

def evaluate(model, loader, device):

    model.eval()

    preds, trues = [], []

    for batch in loader:

        batch = batch.to(device)

        batch = ensure_edge_attr(batch, device)

        out, _ = model(batch)

        preds.append(out.view(-1).cpu())

        trues.append(batch.y.view(-1).cpu())

    preds = torch.cat(preds)

    trues = torch.cat(trues)

    mae = (preds - trues).abs().mean().item()

    rmse = torch.sqrt(((preds - trues) ** 2).mean()).item()

    return mae, rmse

def parse_args():

    p = argparse.ArgumentParser(description="QM9 SDN training (main_qm9)")

    p.add_argument("--csv", default="data/qm9.csv", help="QM9 CSV path")

    p.add_argument("--target", default="Gap", choices=("HOMO", "LUMO", "Gap"))

    p.add_argument(

        "--limit",

        type=int,

        default=None,

        help="Only use the first N molecules (CSV rows) after RDKit filtering. Example: 10000. Default: full dataset.",

    )

    p.add_argument(

        "--ckpt",

        default="best_sdn_qm9_hf.pt",

        help="Checkpoint filename for save/load (under cwd).",

    )

    return parse_args_with_gat_config(p)

def main():

    args = parse_args()

    target = args.target

    path = args.csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_qm9_csv_highfidelity(path, target=target, limit=args.limit)

    lim_note = f" (limit={args.limit})" if args.limit else ""

    print(f"Loaded {len(dataset)} graphs{lim_note}.")

    random.shuffle(dataset)

    n = len(dataset)

    n_train, n_val = int(0.85 * n), int(0.10 * n)

    train_data = dataset[:n_train]

    val_data = dataset[n_train:n_train + n_val]

    test_data = dataset[n_train + n_val:]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_data, batch_size=64)

    test_loader = DataLoader(test_data, batch_size=64)

    in_dim = dataset[0].x.size(-1)

    model = EnhancedOG_PGAT(input_dim=in_dim,

                             hidden=256,

                             n_layers=7,

                             dropout=0.1,

                             num_tasks=1,

                             edge_dim=7).to(device)

    opt = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    sched = CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)

    best_val = 1e9

    patience, patience_counter = 20, 0

    epoch_times = []

    for epoch in range(1, 301):

        t0 = time.perf_counter()

        loss = train_epoch(model, train_loader, opt, device)

        val_mae, val_rmse = evaluate(model, val_loader, device)

        sched.step()

        dt = time.perf_counter() - t0

        epoch_times.append(dt)

        avg_dt = sum(epoch_times) / len(epoch_times)

        print(

            f"Epoch {epoch:03d} | Loss {loss:.6f} | Val MAE {val_mae:.5f} | "

            f"Val RMSE {val_rmse:.5f} | Time {dt:.2f}s | Avg {avg_dt:.2f}s/epoch"

        )

        if val_mae < best_val:

            best_val = val_mae

            patience_counter = 0

            torch.save(model.state_dict(), args.ckpt)

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print("Early stopping!")

                break

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    test_mae, test_rmse = evaluate(model, test_loader, device)

    print(f"\nFINAL TEST MAE: {test_mae:.6f} | RMSE: {test_rmse:.6f}")

    with open("results_qm9_highfidelity.txt", "w") as f:

        f.write(f"Target={target}  Test_MAE={test_mae:.6f}  Test_RMSE={test_rmse:.6f}\n")

if __name__ == "__main__":

    main()
