import os
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable

import torch
from torch_geometric.data import Data, InMemoryDataset

from tdc.single_pred import ADME, Tox
from tdc.metadata import toxicity_dataset_names, adme_dataset_names

from ogb.utils import smiles2graph

from metadata import metrics_to_metrics_fn, metrics_to_loss_fn


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
            data = Tox(name=task_name)
        elif task_name in adme_dataset_names:
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


class ADMEDataset(InMemoryDataset):
    
    def __init__(self,
                 df: pd.DataFrame, 
                 label_cols: list[str], 
                 transform: Optional[Callable] = None):
        """
        df: must have 'smi' column and label columns
        label_cols: list of dataset names or target columns
        transform: optional transform to apply to the data
        """
        super().__init__()

        self.label_cols = label_cols
        self.graphs  = []
        self.targets = []

        for _, row in df.iterrows():
            data_graph_dict = smiles2graph(row['smi'])

            for key, value in data_graph_dict.items():
                if isinstance(value, np.ndarray):
                    data_graph_dict[key] = torch.from_numpy(value)

            data_graph = Data(x=data_graph_dict['node_feat'],
                              edge_index=data_graph_dict['edge_index'],
                              edge_attr=data_graph_dict['edge_feat'])

            if transform is not None:
                data_graph = transform(data_graph)  

            target = row[label_cols].fillna(np.nan).values
            target_tensor = torch.tensor(target, dtype=torch.float)
            
            self.graphs.append(data_graph)
            self.targets.append(target_tensor)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        target = self.targets[idx]
        label_dict = {task: target[i] for i, task in enumerate(self.label_cols)}
        return graph, label_dict
