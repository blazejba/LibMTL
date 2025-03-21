from typing import Any, Dict

import torch
from torch.nn import (ModuleDict,
                      BatchNorm1d, 
                      Embedding,
                      Linear,
                      ModuleList,
                      ReLU,
                      Sequential)

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.data import Data

from losses_and_metrics import SparseCELoss, SparseMSELoss, SparseSpearman


class GPS(torch.nn.Module):

    def __init__(self,
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


def get_decoders(task_dict: Dict[str, Any], channels: int) -> ModuleDict:
    decoders = ModuleDict()
    
    for task, info in task_dict.items():
        if isinstance(info['loss_fn'], SparseCELoss) or isinstance(info['loss_fn'], SparseSpearman):
            decoders[task] = Sequential(
                Linear(channels, channels // 2),
                ReLU(),
                Linear(channels // 2, channels // 4),
                ReLU(),
                Linear(channels // 4, 2)
            )
        elif isinstance(info['loss_fn'], SparseMSELoss):
            decoders[task] = Sequential(
                Linear(channels, channels // 2),
                ReLU(),
                Linear(channels // 2, channels // 4),
                ReLU(),
                Linear(channels // 4, 1)
            )
        else:
            raise NotImplementedError(f"Loss function {info['loss_fn']} not implemented")
    
    return decoders


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
