import os
import glob
import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any

import torch
from torch_sparse import SparseTensor
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from losses_and_metrics import SparseL1, SparseMSELoss
from data import SparseMultitaskDataset

import numpy as np
import pickle as pkl
import torch.nn.functional as F
from tqdm.auto import tqdm
from ogb.utils import smiles2graph


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


def _cast_to_half(data):
    keys = data.keys() if callable(data.keys) else data.keys  # robust to version
    for key in keys:
        val = data[key]
        if torch.is_tensor(val) and val.dtype == torch.float32:
            data[key] = val.half()
    if hasattr(data, "x") and data.x.dtype == torch.float32:
        data.x = data.x.half()
    if hasattr(data, "edge_attr") and data.edge_attr.dtype == torch.float32:
        data.edge_attr = data.edge_attr.half()
    return data


def build_single_graph(row, label_cols, ksteps: int):
    smi = row.smi if hasattr(row, "smi") else row["smi"]

    dg = smiles2graph(smi)
    for k, v in dg.items():
        if isinstance(v, np.ndarray):
            dg[k] = torch.from_numpy(v)

    x = dg["node_feat"].float()
    node_feat_dim = x.size(1)
    edge_index = dg["edge_index"].long()  # [2, E]
    raw_edge_feat = dg["edge_feat"]       # [E, C_edge]

    edge_type_idx = raw_edge_feat.argmax(dim=1)  # [E]
    num_edge_types = raw_edge_feat.size(1)
    edge_one_hot = F.one_hot(edge_type_idx,
                             num_classes=num_edge_types).float()  # [E,C_edge]

    if edge_one_hot.size(1) < node_feat_dim:
        pad = node_feat_dim - edge_one_hot.size(1)
        edge_one_hot = torch.cat(
            [edge_one_hot, torch.zeros(edge_one_hot.size(0), pad)], dim=1
        )
    elif edge_one_hot.size(1) > node_feat_dim:
        edge_one_hot = edge_one_hot[:, :node_feat_dim]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_one_hot,
        edge_weight=torch.ones(edge_index.size(1), dtype=torch.float)
    )

    data = add_full_rrwp(data, walk_length=ksteps)

    data = _cast_to_half(data)

    def _safe_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    if hasattr(row, "_fields"):
        target_arr = np.asarray(
            [_safe_float(getattr(row, col)) for col in label_cols],
            dtype=np.float32,
        )
    else:
        target_arr = row[label_cols].apply(_safe_float).to_numpy(dtype=np.float32)

    target = torch.tensor(target_arr, dtype=torch.float16)

    return data, target


def save_cache(graphs, targets, label_cols, path: str, type: str = "pt"):
    """
    Save a shard to disk.  Exists outside the dataset class so we can build
    shards without keeping everything in RAM.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {"graphs": graphs, "targets": targets, "label_cols": label_cols}
    if type == "pt":
        torch.save(blob, path)
    elif type == "pkl":
        with open(path, "wb") as f:
            pkl.dump(blob, f)
    else:
        raise ValueError(f"Unsupported cache type: {type}")


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

    shard_paths = sorted(glob.glob(os.path.join(cache_dir, "shard_*.pt")))
    if len(shard_paths) < 2:  # we need at least one for train and one for val
        raise ValueError(
            f"Expected ≥2 shard files in {cache_dir}, found {len(shard_paths)}"
        )

    # Use the **last** shard (after sorting) for validation, the rest for training
    val_cache_file = shard_paths[-1]
    train_shard_paths = shard_paths[:50]

    # Build datasets.  Train dataset receives the *list* of shards so it can
    # stream through them; validation gets a single shard.
    train_dataset = SparseMultitaskDataset(
        shard_paths=train_shard_paths,
        ksteps=pe_dim
    )
    val_dataset = SparseMultitaskDataset(
        load_from=val_cache_file,
        ksteps=pe_dim
    )
    return train_dataset, val_dataset



def build_pm6_dataset(
    shard_path: str,
    format: str = "pt",
    ksteps: int = 20,
    cache_dir: Optional[str] = None,
    divide_ratio: int = 4,
) -> None:

    assert cache_dir is not None, "cache_dir must be specified"
    print(f"Processing {shard_path} …")
    df = pd.read_parquet(shard_path)
    df = df.rename(columns={"ordered_smiles": "smi"})

    base_name = os.path.basename(shard_path)

    #   pm6_processed_05.parquet  ->  05
    #   pm6_processed_17.parquet  ->  17
    m = re.search(r'_(\d+)(?=\.[^.]+$)', base_name)
    if m:
        shard_id_num = int(m.group(1))
    else:
        # Fallback: take the very last group of digits in the filename
        digits = re.findall(r'(\d+)', base_name)
        shard_id_num = int(digits[-1]) if digits else 0

    graph_col_names = [name for name in df.columns if "graph_" in name]

    n_total = len(df)
    # Base chunk size (integer division).  Any remainder will be added to
    # the **last** chunk so that we produce exactly `divide_ratio` chunks.
    chunk_size = max(1, n_total // divide_ratio)

    for local_idx in range(divide_ratio):
        start = local_idx * chunk_size
        # The last chunk takes everything that is left (including the remainder)
        if local_idx == divide_ratio - 1:
            sub_df = df.iloc[start:]
        else:
            sub_df = df.iloc[start : start + chunk_size]

        graphs, targets = [], []
        iterator = tqdm(
            sub_df.iterrows(),
            total=len(sub_df),
            desc=f"Building shard {shard_id_num:02d}_{local_idx:02d}",
            disable=False,
        )
        for _, row in iterator:
            g, t = build_single_graph(row, graph_col_names, ksteps)
            graphs.append(g)
            targets.append(t)

        save_path = os.path.join(
            cache_dir, f"shard_{shard_id_num:02d}_{local_idx:02d}.{format}"
        )
        save_cache(graphs, targets, graph_col_names, save_path, type=format)
        print(f"Saved {len(graphs)} samples -> {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--pe-dim", type=int, default=10)
    parser.add_argument("--divide-ratio", type=int, default=4)
    args = parser.parse_args()

    build_pm6_dataset(
        shard_path=args.shard_path,
        cache_dir=args.cache_dir,
        ksteps=args.pe_dim,
        divide_ratio=args.divide_ratio
    )