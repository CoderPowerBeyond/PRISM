#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
| Splitters
"""

import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

__all__ = [
    'RandomSplitter',
    'IndexSplitter',
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
]


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """The abstract class of splitters.
    """

    def __init__(self):
        pass

    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset: List of data.
            frac_train: The proportion of data used for training.
            frac_valid: The proportion of data used for validation.
            frac_test: The proportion of data used for testing.
        """
        raise NotImplementedError


class RandomSplitter(Splitter):
    """Random splitter.
    """

    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset: List of data.
            frac_train: The proportion of data used for training.
            frac_valid: The proportion of data used for validation.
            frac_test: The proportion of data used for testing.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        indices = list(range(N))
        np.random.shuffle(indices)
        train_size = int(N * frac_train)
        valid_size = int(N * frac_valid)
        train_dataset = [dataset[i] for i in indices[:train_size]]
        valid_dataset = [dataset[i] for i in indices[train_size:train_size + valid_size]]
        test_dataset = [dataset[i] for i in indices[train_size + valid_size:]]
        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    """Index splitter.
    """

    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset: List of data.
            frac_train: The proportion of data used for training.
            frac_valid: The proportion of data used for validation.
            frac_test: The proportion of data used for testing.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        train_size = int(N * frac_train)
        valid_size = int(N * frac_valid)
        train_dataset = [dataset[i] for i in range(train_size)]
        valid_dataset = [dataset[i] for i in range(train_size, train_size + valid_size)]
        test_dataset = [dataset[i] for i in range(train_size + valid_size, N)]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    """Scaffold splitter.
    """

    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset: List of data.
            frac_train: The proportion of data used for training.
            frac_valid: The proportion of data used for validation.
            frac_test: The proportion of data used for testing.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        scaffolds = defaultdict(list)
        for i in range(N):
            scaffold = generate_scaffold(dataset[i][0], include_chirality=True)
            scaffolds[scaffold].append(i)
        scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= train_cutoff:
                train_idx.extend(scaffold_set)
            elif len(train_idx) + len(valid_idx) + len(scaffold_set) <= valid_cutoff:
                valid_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
        train_dataset = [dataset[i] for i in train_idx]
        valid_dataset = [dataset[i] for i in valid_idx]
        test_dataset = [dataset[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):
    """Random scaffold splitter.
    """

    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()

    def split(self, dataset, frac_train=None, frac_valid=None, frac_test=None):
        """
        Args:
            dataset: List of data.
            frac_train: The proportion of data used for training.
            frac_valid: The proportion of data used for validation.
            frac_test: The proportion of data used for testing.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)
        scaffolds = defaultdict(list)
        for i in range(N):
            scaffold = generate_scaffold(dataset[i][0], include_chirality=True)
            scaffolds[scaffold].append(i)
        scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
        np.random.shuffle(scaffold_sets)
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in scaffold_sets:
            if len(train_idx) + len(scaffold_set) <= train_cutoff:
                train_idx.extend(scaffold_set)
            elif len(train_idx) + len(valid_idx) + len(scaffold_set) <= valid_cutoff:
                valid_idx.extend(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
        train_dataset = [dataset[i] for i in train_idx]
        valid_dataset = [dataset[i] for i in valid_idx]
        test_dataset = [dataset[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset