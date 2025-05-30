import os
import glob
import pandas as pd
from typing import Optional, Tuple, Callable
from torch_geometric.loader import DataLoader

from losses_and_metrics import SparseL1, SparseMSELoss
from data import SparseMultitaskDataset


def dataloader_factory(
    train_batch_size: int,
    cache_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_pm6_dataset, val_pm6_dataset, test_pm6_dataset = load_pm6_dataset(cache_dir, transform)

    train_loader = DataLoader(
        train_pm6_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        val_pm6_dataset, batch_size=train_batch_size * 2, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_pm6_dataset,
        batch_size=train_batch_size * 2,
        shuffle=False,
        pin_memory=True,
    )

    task_dict = {}
    for task in train_pm6_dataset.label_cols:
        task_dict[task] = {
            "metrics": ["mae"],
            "n_outputs": 1,
            "metrics_fn": SparseL1(),
            "loss_fn": SparseMSELoss("mean"),
            "weight": [0],
        }

    print("PM6 dataloaders ready.")
    return train_loader, valid_loader, test_loader, task_dict


def load_pm6_dataset(
    cache_dir: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> Tuple[SparseMultitaskDataset, SparseMultitaskDataset, SparseMultitaskDataset]:
    shard_paths = [os.path.join(cache_dir, f"pm6_processed_{i:02d}.pt") for i in range(1, 20)]
    val_cache_file = os.path.join(cache_dir, "val_blob.pt")
    test_cache_file = os.path.join(cache_dir, "test_blob.pt")

    if (
        all([os.path.exists(shard_path) for shard_path in shard_paths])
        and os.path.exists(val_cache_file)
        and os.path.exists(test_cache_file)
    ):
        print("Loading PM6 dataloaders from cache...")
        train_dataset = SparseMultitaskDataset(shard_paths=shard_paths, transform=transform)

        val_dataset = SparseMultitaskDataset(load_from=val_cache_file, transform=transform)
        test_dataset = SparseMultitaskDataset(load_from=test_cache_file, transform=transform)
    else:
        raise ValueError(
            "The PM6 dataset is not found in the cache directory. Build it first."
        )
    return train_dataset, val_dataset, test_dataset



def build_pm6_dataset(
    shard_path: str, cache_dir: Optional[str] = None, use_node_features: bool = False
) -> Tuple[SparseMultitaskDataset, SparseMultitaskDataset, SparseMultitaskDataset]:

    print(f"Processing {shard_path}...")
    shard_df = pd.read_parquet(shard_path)
    graph_col_names = [name for name in shard_df.columns if "graph_" in name]
    shard_df = shard_df.rename(columns={"ordered_smiles": "smi"})
    if use_node_features:
        node_col_names = [name for name in shard_df.columns if "node_" in name]
        shard_df = shard_df[graph_col_names + node_col_names + ["smi"]]
    else:
        shard_df = shard_df[graph_col_names + ["smi"]]

    save_path = os.path.join(
        cache_dir, os.path.basename(shard_path).replace(".parquet", ".pt")
    )
    SparseMultitaskDataset(shard_df, graph_col_names, transform=None).save_cache(
        save_path
    )


def combine_pm6_processed_shards(
    cache_dir: str, transform: Optional[Callable] = None, train_frac: float = 0.99
):

    np.random.seed(42)

    shard_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    print(f"Found {len(shard_files)} shard files to combine")

    graphs, targets, label_cols = [], [], None
    for i, _file_path in enumerate(shard_files):
        blob = torch.load(_file_path, map_location="cpu")
        graphs.extend(blob["graphs"])
        targets.extend(blob["targets"])
        if label_cols is None:
            label_cols = blob["label_cols"]
        print(f"Loading blobs {i+1}/{len(shard_files)}")

    print("Shuffling data...")
    val_frac = (1 - train_frac) / 2
    n_samples = len(graphs)
    indices = np.random.permutation(n_samples)

    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_blob = {
        "graphs": [graphs[i] for i in train_indices],
        "targets": [targets[i] for i in train_indices],
        "label_cols": label_cols,
    }
    val_blob = {
        "graphs": [graphs[i] for i in val_indices],
        "targets": [targets[i] for i in val_indices],
        "label_cols": label_cols,
    }
    test_blob = {
        "graphs": [graphs[i] for i in test_indices],
        "targets": [targets[i] for i in test_indices],
        "label_cols": label_cols,
    }

    print("Saving datasets...")
    train_pm6_dataset = SparseMultitaskDataset(
        load_from=train_blob, transform=transform
    )
    val_pm6_dataset = SparseMultitaskDataset(load_from=val_blob, transform=transform)
    test_pm6_dataset = SparseMultitaskDataset(load_from=test_blob, transform=transform)

    train_cache_file = os.path.join(cache_dir, "train_pm6_dataset.pt")
    val_cache_file = os.path.join(cache_dir, "val_pm6_dataset.pt")
    test_cache_file = os.path.join(cache_dir, "test_pm6_dataset.pt")

    train_pm6_dataset.save_cache(train_cache_file)
    val_pm6_dataset.save_cache(val_cache_file)
    test_pm6_dataset.save_cache(test_cache_file)

    print(
        f"Split data into train ({len(train_indices)}), val ({len(val_indices)}), test ({len(test_indices)}) samples"
    )
    print(f"Saved datasets to {cache_dir}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["build", "combine"], default="build"
    )
    parser.add_argument("--shard-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--pe-dim", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "build":
        assert (
            args.shard_path is not None
        ), "Shard path is required for building the dataset"

        build_pm6_dataset(shard_path=args.shard_path, cache_dir=args.cache_dir)
    elif args.mode == "combine":
        import torch
        import numpy as np
        import torch_geometric.transforms as T

        combine_pm6_processed_shards(
            cache_dir=args.cache_dir,
            transform=T.AddRandomWalkPE(walk_length=args.pe_dim, attr_name="pe"),
        )
