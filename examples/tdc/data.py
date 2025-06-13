import os
import time
import contextlib
import threading
import pandas as pd
import pickle as pkl
from tqdm.auto import tqdm
from typing import Dict, Any, List

import torch
from torch_geometric.data import InMemoryDataset

from tdc.single_pred import ADME, Tox
from tdc.metadata import toxicity_dataset_names, adme_dataset_names

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
            'metrics': [metric],
            'n_outputs': 1 if metric in ['mae', 'spearman'] else n_classes - 1,
            'metrics_fn': metrics_to_metrics_fn[metric](),
            'loss_fn': metrics_to_loss_fn[metric](loss_reduction),
            'higher_is_better': metric in ['roc-auc', 'pr-auc']
        }
    return df_train, df_valid, df_test, task_dict


class SparseMultitaskDataset(InMemoryDataset):

    def __init__(
            self,
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

        # background-loading helpers
        self._next_shard_blob: Dict[str, Any] | None = None
        self._preload_thread: threading.Thread | None = None

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
        from examples.tdc.pm6_utils import build_single_graph
        build_fn = lambda row: build_single_graph(row, self.label_cols, self.ksteps)

        self.graphs, self.targets = [], []
        iterator = df.iterrows()
        iterator = tqdm(iterator, total=len(df), desc="Building graphs", disable=False)
        for _, row in iterator:
            g, t = build_fn(row)
            self.graphs.append(g)
            self.targets.append(t)

    def _load_shard_blob(self, path: str) -> Dict[str, Any]:
        "Load a shard file (.pt or .pkl) into memory and return the raw blob."
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pkl.load(f)
        elif path.endswith(".pt"):
            return torch.load(path, map_location="cpu")
        raise ValueError(f"Unsupported shard extension in {path}")
    
    def _apply_blob(self, blob: Dict[str, Any]) -> None:
        "Replace in-memory data with the contents of *blob*."
        self.graphs = blob["graphs"]
        self.targets = blob["targets"]
        self.label_cols = blob["label_cols"]
        if self.transform is not None:
            self.graphs = [self.transform(g) for g in self.graphs]

    def _preload_next(self, next_idx: int) -> None:
        "Background thread target to populate *self._next_shard_blob*."
        blob = self._load_shard_blob(self.shard_paths[next_idx])
        self._next_shard_blob = blob  # atomic replacement is fine for Python objects

    def _start_preload(self, next_idx: int) -> None:
        "Spawn a daemon thread that preloads the shard with index *next_idx*."
        if self._preload_thread and self._preload_thread.is_alive():
            return  # already preloading
        self._preload_thread = threading.Thread(
            target=self._preload_next, args=(next_idx,), daemon=True
        )
        self._preload_thread.start()

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
        # Kick off background preload of the *following* shard
        self._start_preload((shard_idx + 1) % len(self.shard_paths))
        self.current_shard_idx = shard_idx
    
    def __len__(self) -> int:
        # If the previous epoch exhausted this shard, load the next one
        if self.shard_paths and self._reload_flag:
            # Ensure preload is finished
            if self._preload_thread:
                self._preload_thread.join()
            # Swap data
            if self._next_shard_blob is not None:
                self._apply_blob(self._next_shard_blob)
                self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_paths)
                # Start preload of the shard after the one we just swapped in
                self._start_preload((self.current_shard_idx + 1) % len(self.shard_paths))
                self._next_shard_blob = None
            self._samples_seen = 0
            self._reload_flag = False
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
