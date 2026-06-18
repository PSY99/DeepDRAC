"""GraphGPS + GINEConv model for graph-level representation learning.

Uses a GPSConv (Graph Positioning System) with GINEConv backbone,
positional encoding (RandomWalkPE), and global attention pooling
to produce normalized graph embedding vectors.
"""

import torch
import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    LeakyReLU,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch.nn import Sequential as Seq
from torch_geometric.nn import GINEConv, GPSConv, GlobalAttention

from .nn.functional import global_mean_pool_weighted, global_sum_pool_weighted


class NNConv_model(torch.nn.Module):
    def __init__(self, node_features, edge_features, graph_feature, node_out_feature=30, graph_out_feature=2):
        super().__init__()
        torch.manual_seed(12345)
                
        self.num_layers = 2
        self.pe_dim = 2
        self.pe_norm = BatchNorm1d(self.pe_dim)
        self.pe_lin = Linear(self.pe_dim, self.pe_dim)
        self.graph_feature = graph_feature
        self.node_transfomer = Linear(node_features, node_out_feature-self.pe_dim)

        self.convs = ModuleList()
        for _ in range(self.num_layers):
            nn = Sequential(
                Linear(node_out_feature, node_out_feature),
                ReLU(),
                Linear(node_out_feature, node_out_feature),
            )
            self.convs.append(
                GPSConv(node_out_feature, GINEConv(nn,edge_dim=edge_features), heads=1)
            )
        self.act = LeakyReLU(inplace=False)
        self.lin_node = Linear(node_out_feature*2, node_out_feature)

        self.encoder_hidden_size = 32
        self.GlobalAttention = GlobalAttention(
            Seq(Linear(node_out_feature, 1),
                LeakyReLU(inplace=True)),
            Seq(Linear(node_out_feature, node_out_feature),
                LeakyReLU(inplace=True)),
                )
        self.lin_graph = Linear(graph_feature, graph_out_feature)

    def forward(self, batch):
        x, pe, edge_index, edge_attr, graph_attr = batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.graph_attr
        x_pe = self.pe_norm(pe)
        x = self.node_transfomer(x)
        #print("x_shape,x_pe.shape",x.shape,x_pe.shape)
        x = torch.cat((x, self.pe_lin(x_pe)), 1)
        #print("cat_x.shape",x.shape)
        graph_attr = graph_attr.view(-1, self.graph_feature)
        #print("x, edge_index, batch.batch, edge_attr",x,edge_index,batch.batch,edge_attr)
        # 1. Obtain node embeddings
        for conv in self.convs:
            x = conv(x, edge_index, batch.batch, edge_attr=edge_attr)

        # 2. Readout layer
        # embeds_nodes = torch.cat([
        #     global_sum_pool_weighted(x, batch=batch.batch),
        #     global_mean_pool_weighted(x, batch=batch.batch),
        # ], dim=1)
        # x = self.lin_node(embeds_nodes)

        # x = self.act(x)
        x = self.GlobalAttention(x,batch=batch.batch)
        
        graph_encode = self.lin_graph(graph_attr)
        graph_encode = self.act(graph_encode)
        y_vector = torch.cat((x, graph_encode),dim=1)

        # 3. nornmal apply
        y_vector_normalized = F.normalize(y_vector, p=2, dim=1)
        
        return y_vector_normalized

