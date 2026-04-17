

"""
main_pdbbind.py

PDBBind regression using 3D GNNs (SchNet, DimeNet++, MACE).
Input layout (PDBBind core):
  data/v2013-core/
    pdbbind_v2013_core.csv   # columns: pdb_id,label
    <pdb_id>/<pdb_id>_ligand.sdf
    <pdb_id>/<pdb_id>_pocket.pdb
"""

import argparse

from utils.config_utils import parse_args_with_gat_config

import os

import random

import time

from typing import List, Optional, Tuple

import numpy as np

import pandas as pd

from rdkit import Chem

from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch

import torch.nn as nn

from torch_geometric.data import Data

from torch_geometric.loader import DataLoader

from torch_geometric.nn import radius_graph

def set_seed(seed: int = 42) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def safe_element(symbol: str) -> int:

    table = {

        "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,

        "CL": 17, "BR": 35, "I": 53, "ZN": 30, "MG": 12, "CA": 20, "FE": 26

    }

    key = symbol.strip().upper()

    return table.get(key, 0)

def atom_feature_vector(atomic_num: int, is_ligand: int) -> torch.Tensor:

    return torch.tensor(

        [

            float(atomic_num),

            float(atomic_num == 6),

            float(atomic_num == 7),

            float(atomic_num == 8),

            float(atomic_num == 16),

            float(is_ligand),

        ],

        dtype=torch.float,

    )

def parse_pocket_pdb(pocket_pdb: str, max_atoms: int = 1200) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:

    feats: List[torch.Tensor] = []

    coords: List[List[float]] = []

    try:

        with open(pocket_pdb, "r", encoding="utf-8", errors="ignore") as f:

            for line in f:

                if not (line.startswith("ATOM") or line.startswith("HETATM")):

                    continue

                try:

                    x = float(line[30:38].strip())

                    y = float(line[38:46].strip())

                    z = float(line[46:54].strip())

                except ValueError:

                    continue

                elem = line[76:78].strip()

                if not elem:

                    atom_name = line[12:16].strip()

                    elem = atom_name[:1]

                znum = safe_element(elem)

                if znum <= 0:

                    continue

                coords.append([x, y, z])

                feats.append(atom_feature_vector(znum, is_ligand=0))

    except OSError:

        return None

    if len(coords) == 0:

        return None

    pos = torch.tensor(coords, dtype=torch.float)

    x_feat = torch.stack(feats)

    if pos.size(0) > max_atoms:

        center = pos.mean(dim=0, keepdim=True)

        dist = (pos - center).norm(dim=-1)

        keep = torch.topk(dist, k=max_atoms, largest=False).indices

        pos = pos[keep]

        x_feat = x_feat[keep]

    return x_feat, pos

def parse_ligand_sdf(ligand_sdf: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:

    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)

    if suppl is None or len(suppl) == 0:

        return None

    mol = suppl[0]

    if mol is None or mol.GetNumConformers() == 0:

        return None

    conf = mol.GetConformer()

    feats: List[torch.Tensor] = []

    coords: List[List[float]] = []

    for atom in mol.GetAtoms():

        idx = atom.GetIdx()

        p = conf.GetAtomPosition(idx)

        coords.append([p.x, p.y, p.z])

        feats.append(atom_feature_vector(atom.GetAtomicNum(), is_ligand=1))

    x_feat = torch.stack(feats)

    pos = torch.tensor(coords, dtype=torch.float)

    return x_feat, pos

class PDBBindComplexDataset(torch.utils.data.Dataset):

    def __init__(

        self,

        root: str,

        csv_file: str = "pdbbind_v2013_core.csv",

        max_pocket_atoms: int = 1200,

        verbose: bool = True,

        cutoff: float = 5.0,

    ):

        self.samples: List[Data] = []

        labels = pd.read_csv(os.path.join(root, csv_file))

        dropped = 0

        for _, r in labels.iterrows():

            pdb_id = str(r["pdb_id"]).lower()

            y = float(r["label"])

            base = os.path.join(root, pdb_id)

            ligand_sdf = os.path.join(base, f"{pdb_id}_ligand.sdf")

            pocket_pdb = os.path.join(base, f"{pdb_id}_pocket.pdb")

            if not (os.path.exists(ligand_sdf) and os.path.exists(pocket_pdb)):

                dropped += 1

                continue

            lig = parse_ligand_sdf(ligand_sdf)

            poc = parse_pocket_pdb(pocket_pdb, max_atoms=max_pocket_atoms)

            if lig is None or poc is None:

                dropped += 1

                continue

            lig_x, lig_pos = lig

            poc_x, poc_pos = poc

            if lig_x.size(0) < 2 or poc_x.size(0) < 2:

                dropped += 1

                continue

            pos = torch.cat([lig_pos, poc_pos], dim=0)

            x = torch.cat([lig_x, poc_x], dim=0)

            z = x[:, 0].long()

            edge_index = radius_graph(pos, r=cutoff, max_num_neighbors=32)

            data = Data(

                x=x,

                z=z,

                pos=pos,

                edge_index=edge_index,

                y=torch.tensor([y], dtype=torch.float),

                pdb_id=pdb_id

            )

            self.samples.append(data)

        if verbose:

            print(

                f"Loaded PDBBind core: {len(self.samples)} usable complexes "

                f"(dropped {dropped}) from {len(labels)} rows."

            )

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:

        return self.samples[idx]

def split_indices(n: int, seed: int = 42, frac_train: float = 0.8, frac_val: float = 0.1):

    rng = np.random.default_rng(seed)

    idx = np.arange(n)

    rng.shuffle(idx)

    n_train = int(n * frac_train)

    n_val = int(n * frac_val)

    tr = idx[:n_train].tolist()

    va = idx[n_train:n_train + n_val].tolist()

    te = idx[n_train + n_val:].tolist()

    return tr, va, te

def count_parameters(model: nn.Module) -> int:

    """Calculates the total number of trainable parameters in the model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:

    """Evaluates the model and calculates average inference time per sample."""

    model.eval()

    y_true, y_pred = [], []

    total_inference_time = 0.0

    num_samples = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            if device.type == 'cuda':

                torch.cuda.synchronize()

            start_time = time.perf_counter()

            out, _ = model(batch)

            if device.type == 'cuda':

                torch.cuda.synchronize()

            end_time = time.perf_counter()

            total_inference_time += (end_time - start_time)

            num_samples += batch.y.size(0)

            y_true.append(batch.y.cpu().numpy())

            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).reshape(-1)

    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    avg_time_ms = (total_inference_time / num_samples) * 1000 if num_samples > 0 else 0.0

    return mae, rmse, avg_time_ms

def get_model(args, input_dim: int):

    model_type = args.model_type.lower()

    if model_type == "schnet":

        from model.schnet_pg import build_schnet_for_geomo

        return build_schnet_for_geomo(

            input_dim=input_dim, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout

        )

    elif model_type == "dimenet":

        from model.dimenet_pg import build_dimenet_for_geomo

        return build_dimenet_for_geomo(

            input_dim=input_dim, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout

        )

    elif model_type == "mace":

        from model.mace import build_mace_for_geomo

        return build_mace_for_geomo(

            input_dim=input_dim, hidden=args.hidden, n_layers=args.layers, dropout=args.dropout

        )

    else:

        raise ValueError(f"Unknown model type: {model_type}")

def main():

    parser = argparse.ArgumentParser(description="Train 3D GNN on PDBBind core (complex graph).")

    parser.add_argument("--data_root", type=str, default="data/v2013-core")

    parser.add_argument("--label_csv", type=str, default="pdbbind_v2013_core.csv")

    parser.add_argument("--model_type", type=str, default="schnet", choices=["schnet", "dimenet", "mace"], help="Choose the 3D GNN model")

    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--layers", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_curv", type=float, default=1e-4)

    parser.add_argument("--max_pocket_atoms", type=int, default=1200)

    parser.add_argument("--save_ckpt", type=str, default="checkpoints/pdbbind_core_3dgnn.pt")

    args = parse_args_with_gat_config(parser)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)

    dataset = PDBBindComplexDataset(

        root=args.data_root,

        csv_file=args.label_csv,

        max_pocket_atoms=args.max_pocket_atoms,

    )

    if len(dataset) < 30:

        raise RuntimeError(f"Too few usable samples: {len(dataset)}")

    tr_idx, va_idx, te_idx = split_indices(len(dataset), seed=args.seed)

    tr_ds = [dataset[i] for i in tr_idx]

    va_ds = [dataset[i] for i in va_idx]

    te_ds = [dataset[i] for i in te_idx]

    print(f"Split sizes: train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = tr_ds[0].x.size(-1)

    model = get_model(args, input_dim).to(device)

    num_params = count_parameters(model)

    print(f"Model: {args.model_type.upper()}")

    print(f"Total trainable parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.L1Loss()

    best_val = float("inf")

    best_state = None

    for ep in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0

        n_items = 0

        for batch in tr_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            out, _ = model(batch)

            loss = criterion(out.view(-1), batch.y.view(-1)) + args.lambda_curv * model.get_curvature_reg()

            loss.backward()

            optimizer.step()

            bs = batch.y.size(0)

            total_loss += float(loss.item()) * bs

            n_items += bs

        train_loss = total_loss / max(1, n_items)

        val_mae, val_rmse, val_time = evaluate(model, va_loader, device)

        if val_mae < best_val:

            best_val = val_mae

            best_state = {

                "model": model.state_dict(),

                "args": vars(args),

                "best_val_mae": best_val,

            }

            torch.save(best_state, args.save_ckpt)

        print(

            f"[Epoch {ep:03d}] train_loss={train_loss:.4f} "

            f"val_mae={val_mae:.4f} val_rmse={val_rmse:.4f} "

            f"val_inf_time={val_time:.2f}ms/sample best_val={best_val:.4f}"

        )

    if best_state is None:

        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(torch.load(args.save_ckpt, map_location=device)["model"])

    test_mae, test_rmse, test_time = evaluate(model, te_loader, device)

    print(f"Test MAE={test_mae:.4f}  RMSE={test_rmse:.4f}")

    print(f"Average Inference Time: {test_time:.2f} ms / sample")

    print(f"Checkpoint saved to: {args.save_ckpt}")

if __name__ == "__main__":

    main()
