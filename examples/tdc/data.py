import os
import contextlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, Callable, List

import torch
from torch_geometric.data import Data, InMemoryDataset
from tdc.single_pred import ADME, Tox
from tdc.metadata import toxicity_dataset_names, adme_dataset_names

from ogb.utils import smiles2graph

try:
    from rdkit import RDLogger
    from rdkit import Chem
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

from metadata import metrics_to_metrics_fn, metrics_to_loss_fn


def remove_overlap(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame
        ) -> pd.DataFrame:
    
    to_remove = []
    if "[" in df_train['smi'].values[0]:
        for i, mol_row in enumerate(df_train['smi']):

            if i % 100_000 == 0:
                print(f"Removing overlap {i+1}/{len(df_train)}")
            
            mol = Chem.MolFromSmiles(mol_row)
            
            if mol is None:
                continue

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            smi = Chem.MolToSmiles(mol)
            if smi in df_test['smi'].values:
                to_remove.append(smi)
    
    smi_prevent_leakage = set(df_test['smi'])
    df_train = df_train[~df_train['smi'].isin(smi_prevent_leakage)].reset_index(drop=True)
    return df_train


def load_data(datasets_to_use: Dict[str, str],
              loss_reduction: str = 'mean') -> tuple[
                                                        pd.DataFrame, 
                                                        pd.DataFrame, 
                                                        pd.DataFrame, 
                                                        Dict[str, Dict[str, Any]]
                                                   ]:
    
    df_train, df_valid, df_test = None, None, None
    task_dict = {}

    for task_name, metric in datasets_to_use.items():

        if task_name in toxicity_dataset_names:
            with contextlib.redirect_stderr(open(os.devnull,'w')):
                data = Tox(name=task_name)
        elif task_name in adme_dataset_names:
            with contextlib.redirect_stderr(open(os.devnull,'w')):
                data = ADME(name=task_name)
        else:
            raise ValueError(f"Dataset {task_name} not found")

        splits: Dict[str, pd.DataFrame] = data.get_split()

        train = splits['train'][['Drug', 'Y']].copy()
        valid = splits['valid'][['Drug', 'Y']].copy()
        test  = splits['test' ][['Drug', 'Y']].copy()
        
        task_name_lower = task_name.lower()
        train.rename(columns={'Drug': 'smi', 'Y': task_name_lower}, inplace=True)
        valid.rename(columns={'Drug': 'smi', 'Y': task_name_lower}, inplace=True)
        test.rename(columns= {'Drug': 'smi', 'Y': task_name_lower}, inplace=True)
        
        if df_train is None:
            df_train, df_valid, df_test = train, valid, test
        else:
            df_train = pd.merge(df_train, train, on='smi', how='outer')
            df_valid = pd.merge(df_valid, valid, on='smi', how='outer')
            df_test  = pd.merge(df_test, test,   on='smi', how='outer')
        
        n_classes = pd.unique(df_train[task_name_lower]).shape[0]
        task_dict[task_name_lower] = {
            'metrics'    : [metric],
            'n_outputs'  : 1 if metric in ['mae', 'spearman'] else n_classes - 1,
            'metrics_fn' : metrics_to_metrics_fn[metric](),
            'loss_fn'    : metrics_to_loss_fn[metric](loss_reduction),
            'weight'     : [0] if metric in ['mae'] else [1]
        }
    return df_train, df_valid, df_test, task_dict
    

class SparseMultitaskDataset(InMemoryDataset):

    def __init__(self,
                 df: pd.DataFrame | None = None,
                 label_cols: list[str] | None = None,
                 transform: Optional[Callable] = None,
                 load_from: str | Dict | None = None,
                 shard_paths: List[str] | None = None):
        
        super().__init__()

        self.transform = transform

        # --- Sharding support ---
        self.shard_paths: List[str] | None = shard_paths
        self.current_shard_idx: int = 0
        self._reload_flag: bool = False
        self._samples_seen: int = 0

        # If shard paths are provided, load the first shard and exit
        if self.shard_paths:
            # label_cols will be set from the cached shard
            self.load_shard(self.current_shard_idx)
            return

        if load_from is not None:
            self.load_from_cache(load_from)
            return

        assert df is not None, "df must be provided if load_from is not provided"
        assert label_cols is not None, "label_cols must be provided if load_from is not provided"

        self.label_cols = label_cols
        build_fn = self._build_single

        self.graphs, self.targets = [], []
        iterator = df.iterrows()
        iterator = tqdm(iterator, total=len(df), desc="Building graphs", disable=False)
        for _, row in iterator:
            g, t = build_fn(row, transform)
            self.graphs.append(g)
            self.targets.append(t)

    def load_from_cache(self, load_from: str | Dict) -> None:

        if isinstance(load_from, str):
            assert os.path.exists(load_from), f"File {load_from} does not exist"
            blob = torch.load(load_from, map_location="cpu")
        elif isinstance(load_from, dict):
            blob = load_from
        else:
            raise ValueError(f"load_from must be a string or a dictionary, got {type(load_from)}")

        self.graphs = blob["graphs"]
        self.targets = blob["targets"]
        self.label_cols = blob["label_cols"]

        if self.transform is not None:
            self.graphs = [self.transform(g) for g in self.graphs]
        return 

    def load_shard(self, shard_idx: int) -> None:
        """
        Load a shard saved with `save_cache`. The shard index is taken
        modulo the number of shards so cycling past the last shard wraps
        back to the first one.
        """
        if not self.shard_paths:
            raise ValueError("No shard paths provided for sharded dataset.")
        shard_idx = shard_idx % len(self.shard_paths)
        self.load_from_cache(self.shard_paths[shard_idx])
        self.current_shard_idx = shard_idx
    
    def save_cache(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "graphs": self.graphs,
                "targets": self.targets,
                "label_cols": self.label_cols
            },
            path,
        )

    def __len__(self) -> int:
        # If the previous epoch exhausted this shard, load the next one
        if self.shard_paths and self._reload_flag:
            next_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
            self.load_shard(next_idx)
            self._samples_seen = 0
            self._reload_flag = False
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        target = self.targets[idx]

        # --- Sharding bookkeeping ---
        if self.shard_paths:
            self._samples_seen += 1
            if self._samples_seen >= len(self.graphs):
                # All samples from the current shard have been used;
                # flag for reload on the next epoch.
                self._reload_flag = True

        return graph, {task: target[i] for i, task in enumerate(self.label_cols)}

    def _build_single(self, row, transform):
        smi = row.smi if hasattr(row, 'smi') else row['smi']

        dg = smiles2graph(smi)
        for k, v in dg.items():
            if isinstance(v, np.ndarray):
                dg[k] = torch.from_numpy(v)

        data = Data(
            x=dg["node_feat"],
            edge_index=dg["edge_index"],
            edge_attr=dg["edge_feat"],
        )
        if transform is not None:
            data = transform(data)

        if hasattr(row, '_fields'):
            target_vals = [getattr(row, col) for col in self.label_cols]
        else:
            target_vals = row[self.label_cols].fillna(np.nan).values

        target = torch.tensor(target_vals, dtype=torch.float)
        return data, target
