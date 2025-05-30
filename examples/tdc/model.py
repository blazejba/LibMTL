import copy
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (ModuleDict, BatchNorm1d, Embedding,
                      Linear, ModuleList, ReLU, Dropout, Sequential)

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.data import Data

from losses_and_metrics import SparseCELoss, SparseMSELoss, SparseSpearman


class GPS(torch.nn.Module):

    def __init__(
            self,
            channels: int,
            pe_dim: int, 
            num_layers: int,
            attn_type: str,
            attn_kwargs: Dict[str, Any]):
        
        super().__init__()

        self.node_lin = Linear(9, channels - pe_dim)     
        self.pe_lin   = Linear(20, pe_dim)
        self.pe_norm  = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)

        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(Linear(channels, channels), 
                            ReLU(),  
                            Linear(channels, channels))
            conv = GPSConv(channels=channels, 
                           conv=GINEConv(nn), 
                           heads=4,
                           attn_type=attn_type, 
                           attn_kwargs=attn_kwargs)
            self.convs.append(conv)        

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x.float()

        x_pe = self.pe_norm(data.pe)
        x_emb = self.node_lin(x)
        pe_emb = self.pe_lin(x_pe)
        x = torch.cat((x_emb, pe_emb), 1)
        
        edge_attr_indices = torch.argmax(data.edge_attr, dim=1)
        edge_attr_emb = self.edge_emb(edge_attr_indices)
        
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=edge_attr_emb)
        return global_add_pool(x, data.batch)




class GritTransformerLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int, 
            n_heads: int, 
            dropout: float, 
            expansion_factor: int = 2):
        super().__init__()
        assert hidden_dim % n_heads == 0, "Hidden dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ff_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        ff_hidden_dim = hidden_dim * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden_dim, hidden_dim)
        )

    def forward(self, x, attn_bias):
        """
        Forward pass for the transformer layer.
        :param x: Node feature matrix of shape [N, hidden_dim]
        :param attn_bias: Attention bias tensor of shape [n_heads, N, N] (or [N, N] broadcastable)
        """
        N, d = x.size()
        H = self.n_heads

        Q = self.W_q(x).reshape(N, H, self.head_dim).permute(1, 0, 2)
        K = self.W_k(x).reshape(N, H, self.head_dim).permute(1, 0, 2)
        V = self.W_v(x).reshape(N, H, self.head_dim).permute(1, 0, 2)

        scores = torch.einsum('h i d, h j d -> h i j', Q, K)
        scores = scores / (self.head_dim ** 0.5)

        if attn_bias is not None:
            scores += attn_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.einsum('h i j, h j d -> h i d', attn_weights, V)
        out = out.permute(1, 0, 2).reshape(N, d)
        out = self.W_out(out)
        out = self.attn_dropout(out)

        x = self.norm1(x + out)

        ff_out = self.ffn(x)
        ff_out = self.ff_dropout(ff_out)
        return self.norm2(x + ff_out)

class GRIT(nn.Module):
    """
    Graph Inductive Bias Transformer (GRIT)
    """
    
    def __init__(
            self,
            num_node_features: int,
            num_edge_types: int, 
            hidden_dim: int, 
            num_layers: int, 
            num_heads: int, 
            dropout: float = 0.0, 
            rpe_steps: int = 4, 
            max_degree: int = 100):
        """
        :param num_node_features: Dimension of input node features (assumes categorical).
        :param num_edge_types: Number of distinct edge types (for edge embedding bias).
        :param hidden_dim: Hidden dimension size (embedding size for nodes and model width).
        :param num_layers: Number of Transformer layers.
        :param num_heads: Number of attention heads in each layer.
        :param dropout: Dropout rate for attention and feedforward layers.
        :param rpe_steps: K (max number of random walk steps for positional encoding).
        :param max_degree: Max degree to consider for degree embeddings (degrees beyond this are clamped).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_degree = max_degree
        self.rpe_steps = rpe_steps
        self.node_emb = nn.Embedding(num_node_features, hidden_dim)
        self.edge_type_emb = nn.Embedding(num_edge_types, num_heads)
        self.degree_emb = nn.Embedding(max_degree + 1, hidden_dim)

        mlp_in = rpe_steps + 1
        mlp_hidden = hidden_dim // 2 if hidden_dim >= 2 else hidden_dim
        self.rpe_mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_heads)
        )

        self.attn_layers = nn.ModuleList([
            GritTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def _embed_nodes(self, data: Data) -> torch.Tensor:
        """Convert raw node features to learned embeddings."""
        x = data.x
        if x.dtype == torch.long and x.dim() == 1:
            return self.node_emb(x)
        node_idx = torch.argmax(x, dim=1)
        return self.node_emb(node_idx)

    def _compute_degrees(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Return clamped degree for each node."""
        row = edge_index[0]
        deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.long))
        return deg.clamp(max=self.max_degree)

    def _construct_attention_bias(self, edge_index, edge_attr, batch, deg) -> torch.Tensor:
        """Build the per-head attention-bias tensor (RRWP + edge-type)."""
        H = self.num_heads
        N = deg.size(0)
        device = deg.device
        attn_bias = torch.zeros((H, N, N), device=device)
        batch_size = int(batch.max().item()) + 1

        start_idx = 0
        for graph_id in range(batch_size):
            mask = (batch == graph_id)
            idx = mask.nonzero(as_tuple=True)[0]
            n_nodes = idx.size(0)
            if n_nodes == 0:
                continue
            local_idx_map = {int(idx[j]): j for j in range(n_nodes)}

            T = torch.zeros((n_nodes, n_nodes), device=device)
            sub_edge_mask = (batch[edge_index[0]] == graph_id)
            sub_edge_index = edge_index[:, sub_edge_mask]
            sub_edge_attr = edge_attr[sub_edge_mask] if edge_attr is not None else None

            for e_idx, (src, dst) in enumerate(sub_edge_index.t().tolist()):
                if src not in local_idx_map or dst not in local_idx_map:
                    continue
                u = local_idx_map[src]
                v = local_idx_map[dst]
                deg_u = max(1, int(deg[src].item()))
                T[u, v] += 1.0 / deg_u

                if (batch[dst] == graph_id):
                    deg_v = max(1, int(deg[dst].item()))
                    T[v, u] += 1.0 / deg_v

                if self.edge_type_emb is not None and sub_edge_attr is not None:
                    if sub_edge_attr.dim() == 1:
                        et_val = int(sub_edge_attr[e_idx].item())
                    else:
                        et_val = int(sub_edge_attr[e_idx].argmax().item())
                    edge_bias_vec = self.edge_type_emb.weight[et_val]
                    attn_bias[:, start_idx + u, start_idx + v] += edge_bias_vec
                    attn_bias[:, start_idx + v, start_idx + u] += edge_bias_vec

            K = self.rpe_steps
            I = torch.eye(n_nodes, device=device)
            walk_powers = [I]
            if K >= 1:
                walk_powers.append(T)
            current = T
            for _ in range(2, K + 1):
                current = current @ T
                walk_powers.append(current)
            walk_tensor = torch.stack(walk_powers, dim=-1)  # [n_nodes, n_nodes, K+1]
            pair_features = walk_tensor.view(n_nodes * n_nodes, K + 1)
            pair_bias = self.rpe_mlp(pair_features)
            pair_bias = pair_bias.view(n_nodes, n_nodes, H).permute(2, 0, 1)
            attn_bias[:, start_idx:start_idx + n_nodes, start_idx:start_idx + n_nodes] += pair_bias

            start_idx += n_nodes

        if batch_size > 1:
            cum_nodes = [0] + torch.bincount(batch).tolist()
            cum_nodes = torch.cumsum(torch.tensor(cum_nodes), dim=0).tolist()
            for i in range(batch_size):
                for j in range(batch_size):
                    if i == j:
                        continue
                    i_start, i_end = cum_nodes[i], cum_nodes[i + 1]
                    j_start, j_end = cum_nodes[j], cum_nodes[j + 1]
                    attn_bias[:, i_start:i_end, j_start:j_end] = float('-inf')
                    attn_bias[:, j_start:j_end, i_start:i_end] = float('-inf')

        return attn_bias

    def forward(self, data: Data) -> torch.Tensor:
        # 1) Node embeddings
        x = self._embed_nodes(data)

        # 2) Node degrees
        N = x.size(0)
        deg = self._compute_degrees(data.edge_index, N)

        # 3) Attention bias (RRWP + edge types)
        attn_bias = self._construct_attention_bias(
            data.edge_index,
            getattr(data, "edge_attr", None),
            data.batch,
            deg
        )

        # 4) Transformer encoder
        for layer in self.attn_layers:
            x = x + self.degree_emb(deg)
            x = layer(x, attn_bias)

        # 5) Pooling
        return global_add_pool(x, data.batch)


def get_decoders(task_dict: Dict[str, Any], 
                 in_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.1) -> ModuleDict:
    
    hidden_sizes = [in_dim] + [hidden_dim] * num_layers
    
    shared_layers = []
    for i in range(num_layers):
        shared_layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        shared_layers.append(ReLU())
        shared_layers.append(Dropout(dropout))
    
    decoders = ModuleDict()
    for task, info in task_dict.items():
        if isinstance(info['loss_fn'], (SparseCELoss)):
            output_size = info['n_outputs']
        elif isinstance(info['loss_fn'], (SparseMSELoss, SparseSpearman)):
            output_size = 1
        else:
            raise NotImplementedError(f"Loss function {info['loss_fn']} not implemented")
            
        decoders[task] = Sequential(
            *copy.deepcopy(shared_layers), Linear(hidden_sizes[-1], output_size)
        )
    
    return decoders

def freeze_encoder(model):
    dec_param_ids = set()
    if hasattr(model, "decoders"):
        for head in model.decoders.values():
            for p in head.parameters():
                dec_param_ids.add(id(p))

    for name, module in model.named_children():
        if name == "decoders":
            continue

        for p in module.parameters():
            if id(p) not in dec_param_ids:
                p.requires_grad = False

        module.eval()

        orig_fwd = module.forward
        def fwd_no_grad(*args, _orig_fwd=orig_fwd, **kw):
            with torch.no_grad():
                return _orig_fwd(*args, **kw)
        module.forward = fwd_no_grad

    return model

def copy_encoder_weights(src_model, tgt_model):
    src_state = src_model.state_dict()
    filtered_state = {k: v for k, v in src_state.items()
                      if not k.startswith("decoders.")}
    tgt_model.load_state_dict(filtered_state, strict=False)
    return tgt_model




if __name__ == '__main__':

    model = GPS(channels=64, 
                pe_dim=20, 
                num_layers=3, 
                attn_type='multihead', 
                attn_kwargs={'dropout': 0.5})
    decoders = get_decoders({'Caco2_Wang': {}}, 64)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(sum(p.numel() for p in decoder.parameters() if p.requires_grad)
                                       for decoder in decoders.values())
    
    print(f"Trunk trainable params: {total_params:,}")
    print(f"Decoder trainable params: {decoder_params:,}")
