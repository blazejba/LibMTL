import os
import glob
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from torch_geometric.loader import DataLoader

from losses_and_metrics import SparseL1, SparseMSELoss
from data import SparseMultitaskDataset


def dataloader_factory(
    train_batch_size: int,
    cache_dir: Optional[str] = None,
    pe_dim: int = 16
) -> Tuple[DataLoader, DataLoader, int, int, Dict[str, Dict[str, Any]]]:

    train_ds, val_ds = load_pm6_dataset(cache_dir, pe_dim)

    sample_graph, _ = train_ds[0]
    node_dim = sample_graph.x.size(1)
    edge_dim = sample_graph.edge_attr.size(1)

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    valid_loader = DataLoader(val_ds, batch_size=train_batch_size * 2, shuffle=False, pin_memory=True)

    task_dict = {}
    for task in train_ds.label_cols:
        task_dict[task] = {
            "metrics": ["mae"],
            "n_outputs": 1,
            "metrics_fn": SparseL1(),
            "loss_fn": SparseMSELoss("mean"),
            "weight": [0],
        }

    print("PM6 dataloaders ready.")

    return train_loader, valid_loader, node_dim, edge_dim, task_dict


def load_pm6_dataset(
    cache_dir: Optional[str] = None,
    pe_dim: int = 16
) -> Tuple[SparseMultitaskDataset, SparseMultitaskDataset]:
    shard_paths = [os.path.join(cache_dir, f"shard_{i:02d}.pkl") for i in range(0, 74)]
    val_cache_file = os.path.join(cache_dir, "val_blob.pkl")

    if (
        all([os.path.exists(shard_path) for shard_path in shard_paths])
        and os.path.exists(val_cache_file)
    ):
        print("Loading PM6 dataloaders from cache...")
        train_dataset = SparseMultitaskDataset(shard_paths=shard_paths, ksteps=pe_dim)
        val_dataset = SparseMultitaskDataset(load_from=val_cache_file, ksteps=pe_dim)
    else:
        raise ValueError(
            "The PM6 dataset is not found in the cache directory. Build it first."
        )
    return train_dataset, val_dataset



def build_pm6_dataset(
    shard_path: str,
    format: str = "pt",
    ksteps: int = 20,
    cache_dir: Optional[str] = None,
    use_node_features: bool = False
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
        cache_dir, os.path.basename(shard_path).replace(".parquet", f".{format}")
    )
    SparseMultitaskDataset(
        shard_df, graph_col_names, ksteps=ksteps
    ).save_cache(save_path, type="pt")


def split_up_pm6_dataset(
        data_dir: str,
        split_factor: int = 4,
        threshold: int = 1_000_000,
        save_dir: str | None = None,
    ):
    
    if save_dir is None:
        save_dir = data_dir

    shard_files = sorted(glob.glob(os.path.join(data_dir, "pm6_processed_*")))
    shard_idx = 0
    
    for shard_file in shard_files:

        all_paths = [
            os.path.join(save_dir, f"shard_{shard_idx:02d}.pkl")
            for shard_idx in range(shard_idx, shard_idx + split_factor)
        ]
        if all([os.path.exists(path) for path in all_paths]):
            shard_idx += split_factor 
            print(f"Shard {shard_idx} already exists, skipping...")
            continue

        if shard_file.endswith(".pt"):
            shard = torch.load(shard_file, map_location="cpu")
        elif shard_file.endswith(".pkl"):
            with open(shard_file, "rb") as f:
                shard = pkl.load(f)
        else:
            raise ValueError(f"File {shard_file} has an unsupported extension")
        
        n_samples = len(shard['graphs'])
        print(f"{n_samples=}")

        if n_samples < threshold:
            shard_subset = shard
            save_path = os.path.join(save_dir, f"shard_{shard_idx:02d}.pkl")
            print(f"Saving shard {shard_idx} to {save_path}")
            with open(save_path, "wb") as f:
                pkl.dump(shard_subset, f)
            shard_idx += 1
        
        else:
            for i in range(split_factor):
                shard_subset = {
                    'graphs': shard['graphs'][i::split_factor],
                    'targets': shard['targets'][i::split_factor],
                    'label_cols': shard['label_cols'],
                }
                save_path = os.path.join(save_dir, f"shard_{shard_idx:02d}.pkl")
                print(f"Saving shard {shard_idx} to {save_path}")
                with open(save_path, "wb") as f:
                    pkl.dump(shard_subset, f)
                shard_idx += 1


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["build", "split"], default="build"
    )
    parser.add_argument("--shard-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--pe-dim", type=int, default=16)
    args = parser.parse_args()

    if args.mode == "build":
        assert (
            args.shard_path is not None
        ), "Shard path is required for building the dataset"

        build_pm6_dataset(shard_path=args.shard_path, cache_dir=args.cache_dir, ksteps=args.pe_dim)
    elif args.mode == "split":
        import torch
        import pickle as pkl
        split_up_pm6_dataset(
            data_dir=args.cache_dir,
            split_factor=4,
            threshold=1_000_000, 
        )
