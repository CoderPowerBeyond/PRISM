#!/usr/bin/env python3
"""
qm7_dataset.py
==============

Convert original QM7 (`qm7.mat`) dataset into a PyTorch Geometric dataset.

Usage inside main_qm7_sdn.py:
    from qm7_dataset import QM7Dataset
    dataset = QM7Dataset(root="data/QM7")

Structure:
    root/
       qm7.mat
       processed/
           data.pt  (auto-saved)
Dependencies:
    pip install torch torch_geometric scipy numpy
"""

import os
import torch
import numpy as np
import scipy.io as sio
from torch_geometric.data import InMemoryDataset, Data
import torch.nn.functional as F


# ============================================================
# Utility: distance-based bonding
# ============================================================
def build_edges(pos, cutoff=2.0):
    """
    Construct edges based on distance threshold.
    For small organic molecules, cutoff≈2.0 Å works reasonably.
    """
    dist = torch.cdist(pos, pos)
    edge_index = (dist < cutoff).nonzero(as_tuple=False).t().contiguous()
    # remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index


# ============================================================
# Dataset: QM7
# ============================================================
class QM7Dataset(InMemoryDataset):
    """
    Load and preprocess the QM7 dataset stored in qm7.mat.
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        processed_path = os.path.join(self.processed_dir, "data.pt")
        if os.path.exists(processed_path):
            self.data, self.slices = torch.load(processed_path)
        else:
            self._process_and_save(processed_path)

    @property
    def raw_file_names(self):
        return ["qm7.mat"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def _process_and_save(self, save_path):
        raw_path = os.path.join(self.raw_dir, "qm7.mat")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Cannot find {raw_path}\n"
                f"Please place qm7.mat in: {self.raw_dir}/\n"
                f"Download: https://www.quantum-machine.org/data/qm7.mat"
            )

        print(f"Loading QM7 data from {raw_path} ...")
        mat = sio.loadmat(raw_path)

        Z_all = mat["Z"]       # (N, 23)
        R_all = mat["R"]       # (N, 23, 3)
        # Official qm7.mat: atomization energy (DFT), kcal/mol (not Hartree)
        T_all = mat["T"].squeeze()  # (N,)
        N = Z_all.shape[0]

        print(f"Loaded {N} molecules from qm7.mat")

        data_list = []
        for i in range(N):
            # Remove padded zeros (nonexistent atoms)
            mask = Z_all[i] > 0
            Z = Z_all[i][mask]
            R = R_all[i][mask]

            # Convert to PyTorch tensors
            pos = torch.tensor(R, dtype=torch.float)
            atomic_num = torch.tensor(Z, dtype=torch.long)

            # One-hot encode atomic number (Z ≤ 83 for normal organics)
            max_Z = int(Z.max()) if len(Z) > 0 else 0
            x = F.one_hot(atomic_num, num_classes=max(40, max_Z + 1)).float()

            edge_index = build_edges(pos, cutoff=2.0)
            y = torch.tensor([T_all[i]], dtype=torch.float)

            data = Data(x=x, pos=pos, edge_index=edge_index, y=y, z=atomic_num)
            data_list.append(data)

            if (i + 1) % 500 == 0 or i == N - 1:
                print(f"   processed {i + 1}/{N} molecules")

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), save_path)
        self.data, self.slices = data, slices
        print(f"Saved processed dataset to {save_path}")


# ============================================================
# Test run
# ============================================================
if __name__ == "__main__":
    dataset = QM7Dataset(root="data/QM7")
    print(dataset)
    print("Example:", dataset[0])
    print("Num features:", dataset.num_features)
    print("Num graphs:", len(dataset))