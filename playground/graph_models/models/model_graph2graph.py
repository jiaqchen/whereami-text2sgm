import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv, GCNConv
from torch_geometric.nn import aggr, pool
import sys

sys.path.append('/home/julia/Documents/h_coarse_loc')
from playground.graph_models.src.utils import make_cross_graph
        
class SimpleTConv(MessagePassing):
    def __init__(self, in_n, in_e, out_n):
        super().__init__(aggr=aggr.AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(out_n, out_n),
            nn.LeakyReLU(),
            nn.Linear(out_n, out_n),
            nn.LeakyReLU(),
            nn.Linear(out_n, 1)
        )))
        self.TConv = TransformerConv(in_n, out_n, concat=False, heads=2, dropout=0.5)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.TConv(x, edge_index)
        # x = self.propagate(edge_index, x=x)
        x = self.act(x)
        return x

class BigGNN(nn.Module):

    def __init__(self, N):
        super().__init__()
        self.N = N
        in_n, in_e, out_n = 300, 300, 300
        self.TSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n) for _ in range(N)])
        self.GSALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n) for _ in range(N)])
        self.TCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n) for _ in range(N)])
        self.GCALayers = nn.ModuleList([SimpleTConv(in_n, in_e, out_n) for _ in range(N)])

        self.SceneText_MLP = nn.Sequential(
            nn.Linear(300*2, 300), # TODO: input dimension is hardcoded now
            nn.LeakyReLU(),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        self.pooling = pool.SAGPooling(in_channels=out_n, ratio=0.5)


    def forward(self, x_1, x_2,
                      edge_idx_1, edge_idx_2,
                      edge_attr_1, edge_attr_2):
        
        for i in range(self.N):
            ############# Self Attention #############
            x_1 = self.TSALayers[i](x_1, edge_idx_1, edge_attr_1)
            x_2 = self.GSALayers[i](x_2, edge_idx_2, edge_attr_2)
            ############# Self Attention #############

            ############# Cross Attention #############
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2.shape[0]
            edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1.shape, x_2.shape) # First half of x_1_cross should be the original x_1
            edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2.shape, x_1.shape) # First half of x_2_cross should be the original x_2
            x_1_cross = torch.cat((x_1, x_2), dim=0)
            x_2_cross = torch.cat((x_2, x_1), dim=0)
            x_1_cross = self.TCALayers[i](x_1_cross.to('cuda'), edge_index_1_cross.to('cuda'), edge_attr_1_cross.to('cuda'))
            x_2_cross = self.GCALayers[i](x_2_cross.to('cuda'), edge_index_2_cross.to('cuda'), edge_attr_2_cross.to('cuda'))
            x_1 = x_1_cross[:len_x_1] # TODO: Oh, this could be weird....... need to make sure the nodes and indices line up here
            x_2 = x_2_cross[:len_x_2]
            ############# Cross Attention #############
            # x_1 = F.normalize(x_1, p=2, dim=1)
            # x_2 = F.normalize(x_2, p=2, dim=1)
        
        # mean pooling
        x_1_pooled = torch.mean(x_1, dim=0)
        x_2_pooled = torch.mean(x_2, dim=0)

        # Concatenate and feed into SceneTextMLP
        x_concat = torch.cat((x_1_pooled, x_2_pooled), dim=0)
        out_matching = self.SceneText_MLP(x_concat)
        return x_1_pooled, x_2_pooled, out_matching
