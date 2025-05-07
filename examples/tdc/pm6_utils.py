import os
from functools import partial
from typing import Optional, Tuple, Callable

import pandas as pd
from metadata import admet_metrics
from data import load_data, remove_overlap, SparseMultitaskDataset

from torch_geometric.loader import DataLoader

from losses_and_metrics import SparseL1, SparseMSELoss


def dataloader_factory(train_batch_size: int,
                       load_from: Optional[str] = None,
                       transform: Optional[Callable] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print("Building PM6 dataloaders...")

    if load_from:
        train_pm6_dataset = SparseMultitaskDataset(load_from=os.path.join(load_from, 'train_pm6_dataset.pt'))
        val_pm6_dataset = SparseMultitaskDataset(load_from=os.path.join(load_from, 'val_pm6_dataset.pt'))
        test_pm6_dataset = SparseMultitaskDataset(load_from=os.path.join(load_from, 'test_pm6_dataset.pt'))
    else:
        # pm6
        pm6_df = pd.read_parquet('data/pm6/pm6_processed_01.parquet')
        pm6_df = pm6_df.rename(columns={'ordered_smiles': 'smi'})
        graph_col_names = [name for name in pm6_df.columns if "graph_" in name]
        node_col_names = [name for name in pm6_df.columns if "node_" in name]
        pm6_df = pm6_df[graph_col_names + ['smi']]

        # tdc
        _, tdc_val_df, tdc_test_df, _ = load_data(admet_metrics, 'mean')

        # remove overlap
        print(f"PM6 before: {len(pm6_df)}")
        pm6_df = remove_overlap(pm6_df, tdc_test_df)
        pm6_df = remove_overlap(pm6_df, tdc_val_df)
        print(f"PM6 after removing tdc test+val: {len(pm6_df)}")

        # split train/val/test
        train_df = pm6_df.sample(frac=0.98, random_state=42)
        val_df = pm6_df.drop(train_df.index).sample(frac=0.01, random_state=42)
        test_df = pm6_df.drop(train_df.index).drop(val_df.index)

        # build datasets and cache
        partial_dataset = partial(SparseMultitaskDataset, label_cols=graph_col_names, transform=transform)
        train_pm6_dataset = partial_dataset(train_df)
        val_pm6_dataset = partial_dataset(val_df)
        test_pm6_dataset = partial_dataset(test_df)

        # save datasets
        train_pm6_dataset.save_cache('data/pm6/train_pm6_dataset.pt')
        val_pm6_dataset.save_cache('data/pm6/val_pm6_dataset.pt')
        test_pm6_dataset.save_cache('data/pm6/test_pm6_dataset.pt')

    train_loader = DataLoader(train_pm6_dataset, batch_size=train_batch_size, shuffle=True,  pin_memory=True)
    valid_loader = DataLoader(val_pm6_dataset, batch_size=train_batch_size*2, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_pm6_dataset,  batch_size=train_batch_size*2, shuffle=False, pin_memory=True)

    task_dict = {}
    for task in train_pm6_dataset.label_cols:
        task_dict[task] = {
            'metrics'    : ['mae'],
            'n_outputs'  : 1,
            'metrics_fn' : SparseL1(),
            'loss_fn'    : SparseMSELoss('mean'),
            'weight'     : [0]
        }

    print("PM6 dataloaders ready.")
    return train_loader, valid_loader, test_loader, task_dict
