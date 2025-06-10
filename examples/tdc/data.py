import os
import time
import contextlib
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm.auto import tqdm
from typing import Dict, Any, Optional, Callable, List

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
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
                 load_from: str | Dict | None = None,
                 shard_paths: List[str] | None = None,
                 ksteps: int = 20):
        
        super().__init__()

        self.shard_paths: List[str] | None = shard_paths
        self.current_shard_idx: int = 0
        self._reload_flag: bool = False
        self._samples_seen: int = 0
        self.ksteps = ksteps

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
            g, t = build_fn(row)
            self.graphs.append(g)
            self.targets.append(t)

    def load_from_cache(self, load_from: str | Dict) -> None:

        if isinstance(load_from, str):
            assert os.path.exists(load_from), f"File {load_from} does not exist"
            if load_from.endswith(".pkl"):
                with open(load_from, "rb") as f:
                    blob = pkl.load(f)
            elif load_from.endswith(".pt"):
                blob = torch.load(load_from, map_location="cpu")
            else:
                raise ValueError(f"File {load_from} has an unsupported extension")
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
    
    def save_cache(self, path: str, type: str = "pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if type == "pkl":
            with open(path, "wb") as f:
                pkl.dump(
                    {
                        "graphs": self.graphs,
                        "targets": self.targets,
                        "label_cols": self.label_cols
                    }, f)
        elif type == "pt":
            torch.save(
                {
                    "graphs": self.graphs,
                    "targets": self.targets,
                    "label_cols": self.label_cols
                }, path)
        else: 
            raise ValueError(f"Unsupported type: {type}")
            
    def __len__(self) -> int:
        # If the previous epoch exhausted this shard, load the next one
        if self.shard_paths and self._reload_flag:
            print(f"Loading next shard {self.current_shard_idx + 1}")
            start_time = time.time()
            next_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
            self.load_shard(next_idx)
            self._samples_seen = 0
            self._reload_flag = False
            print(f"Loaded shard {next_idx} in {time.time() - start_time:.2f} seconds")
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        target = self.targets[idx]

        # sharding bookkeeping
        if self.shard_paths:
            self._samples_seen += 1
            if self._samples_seen >= len(self.graphs):
                self._reload_flag = True

        return graph, {task: target[i] for i, task in enumerate(self.label_cols)}

    def _build_single(self, row):
        smi = row.smi if hasattr(row, 'smi') else row['smi']

        dg = smiles2graph(smi)
        for k, v in dg.items():
            if isinstance(v, np.ndarray):
                dg[k] = torch.from_numpy(v)

        x = dg["node_feat"].float()
        node_feat_dim = x.size(1)
        edge_index = dg["edge_index"].long() # [2, E]
        raw_edge_feat = dg["edge_feat"] # [E, C_edge]

        edge_type_idx = raw_edge_feat.argmax(dim=1) # [E]
        num_edge_types = raw_edge_feat.size(1)
        edge_one_hot = F.one_hot(edge_type_idx,
                                 num_classes=num_edge_types).float() # [E,C_edge]

        if edge_one_hot.size(1) < node_feat_dim:
            pad = node_feat_dim - edge_one_hot.size(1)
            edge_one_hot = torch.cat(
                [edge_one_hot, torch.zeros(edge_one_hot.size(0), pad)], dim=1
            )
        elif edge_one_hot.size(1) > node_feat_dim:
            edge_one_hot = edge_one_hot[:, :node_feat_dim]

        edge_attr = edge_one_hot # [E, F_node]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        data.edge_weight = torch.ones(
            edge_index.size(1),
            dtype=torch.float,
            device=edge_index.device
        )

        data = add_full_rrwp(data, walk_length=self.ksteps)

        if hasattr(row, '_fields'):
            target_vals = [getattr(row, col) for col in self.label_cols]
        else:
            target_vals = row[self.label_cols].fillna(np.nan).values

        target = torch.tensor(target_vals, dtype=torch.float)

        return data, target


@torch.no_grad()
def add_full_rrwp(data,
                  walk_length=8,
                  attr_name_abs="rrwp",
                  attr_name_rel="rrwp",
                  add_identity=True):
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes))

    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1)  # n x n x k
    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data