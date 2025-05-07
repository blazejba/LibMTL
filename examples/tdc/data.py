import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from tqdm.auto import tqdm
import contextlib

import torch
from torch_geometric.data import Data, InMemoryDataset

from tdc.single_pred import ADME, Tox
from tdc.metadata import toxicity_dataset_names, adme_dataset_names

from ogb.utils import smiles2graph

try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

from metadata import metrics_to_metrics_fn, metrics_to_loss_fn


def remove_overlap(df_target: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
    leakage_smiles = set(df_source['smi'])
    df_target = df_target[~df_target['smi'].isin(leakage_smiles)].reset_index(drop=True)
    return df_target


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
                 load_from: str | None = None):
        
        super().__init__()

        if load_from is not None:
            assert os.path.exists(load_from), f"File {load_from} does not exist"
            
            blob = torch.load(load_from, map_location="cpu")
            self.graphs = blob["graphs"]
            self.targets = blob["targets"]
            self.label_cols = blob["label_cols"]

            if transform is not None:
                self.graphs = [transform(g) for g in self.graphs]
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
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        target = self.targets[idx]
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
