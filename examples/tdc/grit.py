import numpy as np

import torch
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_mean_pool

import opt_einsum as oe


def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class MultiHeadAttentionLayerGritSparse(nn.Module):

    def __init__(
            self, 
            in_dim,
            out_dim,
            num_heads,
            use_bias,
            clamp=5.0,
            dropout=0.0):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
        nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
        score = score * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + E_b

        score = torch.relu(score)
        e_t = score

        batch.wE = score.flatten(1)

        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = scatter(msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce='add')

        rowV = scatter(e_t * score, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="add")
        rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
        batch.wV = batch.wV + rowV
        
        return batch

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        batch = self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


class GritTransformerLayer(nn.Module):

    def __init__(
            self,
            in_dim,
            out_dim,
            num_heads,
            dropout=0.0,
            attn_dropout=0.0,
            clamp=5.0,
            bn_momentum=0.1):
        super().__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            dropout=attn_dropout,
            clamp=clamp,
            use_bias=True
        )

        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
        nn.init.xavier_normal_(self.deg_coef)

        self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)
        self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)
        self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=True, eps=1e-5, momentum=bn_momentum)

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = batch.log_deg
        if log_deg.dim() == 1:
            log_deg = log_deg.unsqueeze(-1)

        h_in1 = h  # for first residual connection
        e_in1 = batch.edge_attr
        
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        h = torch.stack([h, h * log_deg], dim=-1)
        h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        e = e_attn_out.flatten(1)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.O_e(e)

        h = h_in1 + h  # residual connection
        e = e + e_in1

        h = self.batch_norm1_h(h)
        e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h # for second residual connection
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        h = h_in2 + h # residual connection

        h = self.batch_norm2_h(h)

        batch.x = h
        batch.edge_attr = e

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            super().__repr__(),
        )


class GritTransformer(torch.nn.Module):

    def __init__(
            self,
            node_dim,
            edge_dim,
            rrwp_dim,
            hidden_dim,
            num_heads,
            num_layers,
            dropout=0.0,
            attn_dropout=0.0,
            clamp=5.0,
            bn_momentum=0.1,
            ):
        
        super().__init__()

        self.node_proj = nn.Linear(node_dim,  hidden_dim)
        self.edge_proj = nn.Linear(edge_dim,  hidden_dim)
        self.rrwp_node_encoder = RRWPLinearNodeEncoder(emb_dim=rrwp_dim, out_dim=hidden_dim)
        self.rrwp_edge_encoder = RRWPLinearEdgeEncoder(emb_dim=rrwp_dim, out_dim=hidden_dim)

        self.grit_layers = nn.ModuleList([
            GritTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                attn_dropout=attn_dropout,
                clamp=clamp,
                bn_momentum=bn_momentum)
            for _ in range(num_layers)
        ])


    def forward(self, batch):
        batch.x = self.node_proj(batch.x.float())
        batch.edge_attr = self.edge_proj(batch.edge_attr.float())
        batch = self.rrwp_node_encoder(batch)
        batch = self.rrwp_edge_encoder(batch)
        for layer in self.grit_layers:
            batch = layer(batch)
        return global_mean_pool(batch.x, batch.batch)



class RRWPLinearNodeEncoder(torch.nn.Module):
    def __init__(
            self, 
            emb_dim, 
            out_dim, 
            use_bias=False, 
            pe_name="rrwp"):
        
        super().__init__()
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, batch):
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        return batch
    

class RRWPLinearEdgeEncoder(torch.nn.Module):
    def __init__(
            self, 
            emb_dim, 
            out_dim,
            fill_value=0.0):
        
        super().__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim

        self.fc = nn.Linear(emb_dim, out_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)

        out_idx, out_val = torch_sparse.coalesce(
            torch.cat([edge_index, rrwp_idx], dim=1),
            torch.cat([edge_attr, rrwp_val], dim=0),
            batch.num_nodes, batch.num_nodes,
            op="add"
        )

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"
