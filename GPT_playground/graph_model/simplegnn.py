###################################### DATA ######################################

import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F

import copy
from sg_dataloader import SceneGraph



###################################### MODEL ######################################

import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class BigGNN(nn.Module):
    def __init__(self, x_1_dim, x_2_dim):
        super().__init__()
        self.TextSelfAttention = SimpleGNN(600, 300, 300)
        self.GraphSelfAttention = SimpleGNN(600, 300, 300)
        self.TextCrossAttention = SimpleGNN(600, 300, 300)
        self.GraphCrossAttention = SimpleGNN(600, 300, 300)
        self.SceneText_MLP = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU()
        )

        # Make Cross Attention Graphs
        self.edge_index_1_cross, self.edge_attr_1_cross = self.makeCrossGraph(x_1_dim, x_2_dim)
        self.edge_index_2_cross, self.edge_attr_2_cross = self.makeCrossGraph(x_2_dim, x_1_dim)
        

    def forward(self, x_1, x_2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2):
        # Self Attention
        x_1 = self.TextSelfAttention(x_1, edge_index_1, edge_attr_1)
        x_2 = self.GraphSelfAttention(x_2, edge_index_2, edge_attr_2)

        # Length of x_1 and x_2
        len_x_1 = x_1.shape[0]
        len_x_2 = x_2.shape[0]

        # Make Cross Attention Graphs
        # x_1_cross, edge_index_1_cross, edge_attr_1_cross = self.makeCrossGraph(x_1, x_2) # First half of x_1_cross should be the original x_1
        # x_2_cross, edge_index_2_cross, edge_attr_2_cross = self.makeCrossGraph(x_2, x_1) # First half of x_2_cross should be the original x_2

        # Concatenate x_1 and x_2
        x_1_cross = torch.cat((x_1, x_2), dim=0)
        x_2_cross = torch.cat((x_2, x_1), dim=0)

        # Cross Attention
        x_1_cross = self.TextCrossAttention(x_1_cross, self.edge_index_1_cross, self.edge_attr_1_cross)
        x_2_cross = self.GraphCrossAttention(x_2_cross, self.edge_index_2_cross, self.edge_attr_2_cross)

        # Get the first len_x_1 nodes from x_1_cross
        x_1_cross = x_1_cross[:len_x_1]
        x_2_cross = x_2_cross[:len_x_2]

        return x_1_cross, x_2_cross
    
    def makeCrossGraph(self, x_1_dim, x_2_dim):
        x_1_dim = x_1_dim[0]
        x_2_dim = x_2_dim[0]

        edge_index_cross = torch.tensor([[], []], dtype=torch.long)
        edge_attr_cross = torch.tensor([], dtype=torch.float)

        # Add edge from each node in x_1 to x_2
        for i in range(x_1_dim):
            for j in range(x_2_dim):
                edge_index_cross = torch.cat((edge_index_cross, torch.tensor([[i], [x_1_dim + j]], dtype=torch.long)), dim=1)
                # Add edge_attr which is dimension 1x300, all 0
                edge_attr_cross = torch.cat((edge_attr_cross, torch.zeros((1, 300), dtype=torch.float)), dim=0) # TODO: dimension 300

        assert(edge_index_cross.shape[1] == x_1_dim * x_2_dim)
        assert(edge_attr_cross.shape[0] == x_1_dim * x_2_dim)
        return edge_index_cross, edge_attr_cross
    
    # def makeCrossGraph(self, x_1, x_2):
    #     edge_index_cross = torch.tensor([[], []], dtype=torch.long)
    #     edge_attr_cross = torch.tensor([], dtype=torch.float)

    #     # Add edge from each node in x_1 to x_2
    #     for i in range(x_1.shape[0]):
    #         for j in range(x_2.shape[0]):
    #             edge_index_cross = torch.cat((edge_index_cross, torch.tensor([[i], [x_1.shape[0] + j]], dtype=torch.long)), dim=1)
    #             # Add edge_attr which is dimension 1x300, all 0
    #             edge_attr_cross = torch.cat((edge_attr_cross, torch.zeros((1, 300), dtype=torch.float)), dim=0) # TODO: dimension 300

    #     x_1_cross = torch.cat((x_1, x_2), dim=0) # Concatenate all nodes together

    #     assert(x_1_cross.shape[0] == x_1.shape[0] + x_2.shape[0])
    #     assert(edge_index_cross.shape[1] == x_1.shape[0] * x_2.shape[0])
    #     assert(edge_attr_cross.shape[0] == x_1.shape[0] * x_2.shape[0])
    #     return x_1_cross, edge_index_cross, edge_attr_cross

class SimpleGNN(MessagePassing):
    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super(SimpleGNN, self).__init__(aggr='add')  # "add" aggregation
        self.lin_node = torch.nn.Linear(in_channels_node, out_channels)
        self.lin_edge = torch.nn.Linear(in_channels_edge, out_channels)
        self.lin = torch.nn.Linear(out_channels, in_channels_node)

    def forward(self, x, edge_index, edge_attr):
        # x is the input node features and edge_index indicates the graph structure
        projected_x = self.lin_node(x)
        projected_edge_attr = self.lin_edge(edge_attr)
        # return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # also propagate based on node's neighbors
        out = self.propagate(edge_index, x=projected_x, edge_attr=projected_edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        # x_j is the feature of the neighboring node
        # edge_attr is the edge feature
        return x_i + x_j + edge_attr

    def update(self, aggr_out):
        # aggr_out is the output of self.message aggregated according to self.aggr (here, "add")
        return self.lin(aggr_out)
    
###################################### TRAIN ######################################

def mask_node(x, p=0.1):
    # Mask a random row in x with 1's
    x_clone = x.clone()
    num_nodes_to_mask = int(x.shape[0] * p) + 1
    rows_to_mask = torch.randperm(x.shape[0])[:num_nodes_to_mask]
    x_clone[rows_to_mask] = 1
    return x_clone

def train_dummy_big_gnn(graph1, graph2):
    # Nodes
    x_1 = torch.tensor(graph1.get_node_features(), dtype=torch.float)    # Node features
    x_2 = torch.tensor(graph2.get_node_features(), dtype=torch.float)    # Node features

    # Edges
    sources_1, targets_1, features_1 = graph1.get_edge_s_t_feats()
    print("features.shape", features_1.shape)                           # Edge features
    print("nodes.shape", graph1.get_node_features().shape)             # Node features
    assert(len(sources_1) == len(targets_1) == len(features_1))
    edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
    edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

    sources_2, targets_2, features_2 = graph2.get_edge_s_t_feats()
    print("features.shape", features_2.shape)                           # Edge features
    print("nodes.shape", graph2.get_node_features().shape)             # Node features
    assert(len(sources_2) == len(targets_2) == len(features_2))
    edge_index_2 = torch.tensor([sources_2, targets_2], dtype=torch.long)
    edge_attr_2 = torch.tensor(features_2, dtype=torch.float)

    # Mask node
    x_1_masked = mask_node(x_1)
    x_2_masked = mask_node(x_2)
    data1 = Data(x=x_1_masked, edge_index=edge_index_1, edge_attr=edge_attr_1)
    data2 = Data(x=x_2_masked, edge_index=edge_index_2, edge_attr=edge_attr_2)

    model = BigGNN(x_1_masked.shape, x_2_masked.shape) # TODO: input output channels are hardcoded now, need to figure that out
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1000):
        # Training step
        optimizer.zero_grad() # Clear gradients.
        out1, out2 = model(x_1_masked, x_2_masked, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2) # Perform a single forward pass.
        loss1 = F.mse_loss(out1, x_1) # Compute the loss.
        loss2 = F.mse_loss(out2, x_2) # Compute the loss.
        loss = loss1 + loss2
        loss.backward() # Derive gradients.
        optimizer.step() # Update parameters based on gradients.

        # Print loss
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

def train_dummy_simple_gnn(graph):
    # Nodes
    x = torch.tensor(graph.get_node_features(), dtype=torch.float)    # Node features

    # Edges
    sources, targets, features = graph.get_edge_s_t_feats()
    print("features.shape", features.shape)                           # Edge features
    print("nodes.shape", graph.get_node_features().shape)             # Node features
    assert(len(sources) == len(targets) == len(features))
    edge_index = torch.tensor([sources, targets], dtype=torch.long) 
    edge_attr = torch.tensor(features, dtype=torch.float)

    # Mask node
    x_masked = mask_node(x)
    data = Data(x=x_masked, edge_index=edge_index, edge_attr=edge_attr)

    model = SimpleGNN(in_channels_node=600, in_channels_edge=300, out_channels=300)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1000):
        # Training step
        optimizer.zero_grad()  # Clear gradients.
        out = model(x_masked, edge_index, edge_attr)  # Perform a single forward pass.
        loss = F.mse_loss(out, x)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        # Print loss
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

if __name__ == '__main__':
    # Load scene graph
    scene_graph_human = SceneGraph('human+GPT', '0cac75ce-8d6f-2d13-8cf1-add4e795b9b0', raw_json='../output_clean/0cac75ce-8d6f-2d13-8cf1-add4e795b9b0/1_300ms/0_gpt_clean.json')
    scene_graph_3dssg = SceneGraph('3DSSG', '0cac75ce-8d6f-2d13-8cf1-add4e795b9b0', euc_dist_thres=1.0)

    # Process graph such that there are no gaps in indices and all nodes index from 0
    scene_graph_human.to_pyg()
    scene_graph_3dssg.to_pyg()

    # Make Hierarchical node that has an edge connecting to all other nodes
    scene_graph_human.add_place_node()
    scene_graph_3dssg.add_place_node()

    train_dummy_big_gnn(scene_graph_human, scene_graph_3dssg)

    # TODO: Train on a set of 10 graphs
    # TODO: Validate on 10 examples to see if we learn anything