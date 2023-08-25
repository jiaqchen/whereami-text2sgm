###################################### DATA ######################################

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv

import copy
from sg_dataloader import SceneGraph

from utils import print_closest_words, make_cross_graph, mask_node

###################################### MODEL ######################################

class BigGNN(nn.Module):
    def __init__(self, x_1_dim, x_2_dim, place_node_1_idx, place_node_2_idx):
        super().__init__()
        self.TextSelfAttention = SimpleGNN(300, 300, 300)
        self.GraphSelfAttention = SimpleGNN(300, 300, 300)
        self.TextCrossAttention = SimpleGNN(900, 300, 300)
        self.GraphCrossAttention = SimpleGNN(900, 300, 300)
        # MLP for predicting matching score between 0 and 1
        self.SceneText_MLP = nn.Sequential(
            nn.Linear(600, 600), # TODO: input dimension is hardcoded now
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        # self.place_node_1_idx = place_node_1_idx
        # self.place_node_2_idx = place_node_2_idx

        # Make Cross Attention Graphs
        self.edge_index_1_cross, self.edge_attr_1_cross = make_cross_graph(x_1_dim, x_2_dim)
        self.edge_index_2_cross, self.edge_attr_2_cross = make_cross_graph(x_2_dim, x_1_dim)
        

    def forward(self, x_1, x_2, 
                edge_index_1, edge_index_2, 
                edge_attr_1, edge_attr_2, 
                place_node_1_idx=None, place_node_2_idx=None):
        assert(place_node_1_idx is not None) # TODO: hacky? or not? maybe it's fine to set the place_node_idx every iteration
        assert(place_node_2_idx is not None)
        # Change edge_index_1_cross and edge_index_2_cross at test time
        # if (not self.training):
            # self.place_node_1_idx = place_node_1_idx
            # self.place_node_2_idx = place_node_2_idx

        N = 16 # Number of iterations
        for _ in range(N):

            # Batch Norm
            x_1 = F.normalize(x_1, p=2, dim=1)
            x_2 = F.normalize(x_2, p=2, dim=1)

            ############# Self Attention #############
            
            x_1 = self.TextSelfAttention(x_1, edge_index_1, edge_attr_1)
            x_2 = self.GraphSelfAttention(x_2, edge_index_2, edge_attr_2)

            # Length of x_1 and x_2
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2.shape[0]

            ############# Cross Attention #############

            # Make Cross Attention Graphs
            # x_1_cross, edge_index_1_cross, edge_attr_1_cross = self.make_cross_graph(x_1, x_2) # First half of x_1_cross should be the original x_1
            # x_2_cross, edge_index_2_cross, edge_attr_2_cross = self.make_cross_graph(x_2, x_1) # First half of x_2_cross should be the original x_2

            # Concatenate x_1 and x_2
            x_1_cross = torch.cat((x_1, x_2), dim=0)
            x_2_cross = torch.cat((x_2, x_1), dim=0)

            # Cross Attention
            x_1_cross = self.TextCrossAttention(x_1_cross, self.edge_index_1_cross, self.edge_attr_1_cross)
            x_2_cross = self.GraphCrossAttention(x_2_cross, self.edge_index_2_cross, self.edge_attr_2_cross)

            # Get the first len_x_1 nodes from x_1_cross
            x_1 = x_1_cross[:len_x_1]
            x_2 = x_2_cross[:len_x_2]

        ############# MLP #############

        # Find the place node from x_1 and x_2
        place_node_1 = x_1[place_node_1_idx] # TODO: check that place node is from x_1
        place_node_2 = x_2[place_node_2_idx] # TODO: check that place node is from x_2

        # MLP to get a matching score
        matching_score = self.SceneText_MLP(torch.cat((place_node_1, place_node_2), dim=0))

        return x_1, x_2, matching_score

class SimpleGNN(MessagePassing):
    # Simple one layer GATConv
    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super(SimpleGNN, self).__init__(aggr='add')  # "add" aggregation
        self.GATConv_nodes = GATConv(in_channels_node, out_channels, heads=1, dropout=0.5)

    def forward(self, x, edge_index, edge_attr):
        x = self.GATConv_nodes(x, edge_index)
        x = F.relu(x)
        # Normalize
        x = F.normalize(x, p=2, dim=1)
        return x

# class SimpleGNN(MessagePassing):
#     def __init__(self, in_channels_node, in_channels_edge, out_channels):
#         super(SimpleGNN, self).__init__(aggr='add')  # "add" aggregation
#         self.lin_node = torch.nn.Linear(in_channels_node, out_channels)
#         self.lin_edge = torch.nn.Linear(in_channels_edge, out_channels)
#         self.lin = torch.nn.Linear(out_channels, in_channels_node)

#         self.message_lin = torch.nn.Linear(out_channels, out_channels)

#     def forward(self, x, edge_index, edge_attr):
#         # x is the input node features and edge_index indicates the graph structure
#         projected_x = self.lin_node(x)
#         projected_edge_attr = self.lin_edge(edge_attr)
#         # return self.propagate(edge_index, x=x, edge_attr=edge_attr)
#         # also propagate based on node's neighbors
#         out = self.propagate(edge_index, x=projected_x, edge_attr=projected_edge_attr)
#         return out

#     def message(self, x_i, x_j, edge_attr):
#         # x_j is the feature of the neighboring node
#         # edge_attr is the edge feature
#         # x_i is the feature of the central node
        
#         return self.message_lin(x_j)

#     def update(self, aggr_out):
#         # aggr_out is the output of self.message aggregated according to self.aggr (here, "add")
#         out = self.lin(aggr_out)
#         # normalize
#         # out = F.normalize(out, p=2, dim=1)
#         return out
    
###################################### TRAIN ######################################

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

    # Get Place Node Index
    _, place_node_1_idx = graph1.get_place_node_idx()
    _, place_node_2_idx = graph2.get_place_node_idx()

    # Mask node
    x_1_masked = mask_node(x_1)
    x_2_masked = mask_node(x_2)
    data1 = Data(x=x_1_masked, edge_index=edge_index_1, edge_attr=edge_attr_1)
    data2 = Data(x=x_2_masked, edge_index=edge_index_2, edge_attr=edge_attr_2)

    # Normalize input
    x_1_masked = F.normalize(x_1_masked, p=2, dim=1)
    x_2_masked = F.normalize(x_2_masked, p=2, dim=1)

    model = BigGNN(x_1_masked.shape, x_2_masked.shape, place_node_1_idx, place_node_2_idx) # TODO: input output channels are hardcoded now, need to figure that out
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1000):
        # Training step
        optimizer.zero_grad() # Clear gradients.
        out1, out2, out_matching = model(x_1_masked, x_2_masked, 
                                        edge_index_1, edge_index_2, 
                                        edge_attr_1, edge_attr_2, 
                                        place_node_1_idx, place_node_2_idx) # Perform a single forward pass.
        loss1 = F.mse_loss(out1, x_1) # Compute the loss.
        loss2 = F.mse_loss(out2, x_2) # Compute the loss.
        loss3 = F.mse_loss(out_matching, torch.tensor([1.0], dtype=torch.float)) # TODO: loss3 is distance vector, but could also try cosine similarity, or after MLP
        # loss3 = F.mse_loss(out1[place_node_1_idx], out2[place_node_2_idx]) # TODO: loss3 is distance vector, but could also try cosine similarity, or after MLP
        loss = loss1 + loss2 + loss3
        loss.backward() # Derive gradients.
        optimizer.step() # Update parameters based on gradients.

        # Print loss
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # Check accuracy
        if epoch % 499 == 0 and epoch != 0:
            out1_vector = out1.detach().numpy()
            out2_vector = out2.detach().numpy()
            x_1_vector = x_1.detach().numpy()
            x_2_vector = x_2.detach().numpy()
            print_closest_words(out1_vector, x_1_vector)
            # print_closest_words(out2_vector, x_2_vector)

    return model

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

def evaluate_model(model, scene_graph_human, scene_graph_3dssg):
    # Evaluate model on another graph pair
    model.eval()
    with torch.no_grad():
        # Process graph such that there are no gaps in indices and all nodes index from 0
        scene_graph_human.to_pyg()
        scene_graph_3dssg.to_pyg()

        # Make Hierarchical node that has an edge connecting to all other nodes
        scene_graph_human.add_place_node() # TODO: this method should return the place_node already
        scene_graph_3dssg.add_place_node()

        # Get x_1 and x_2
        x_1 = torch.tensor(scene_graph_human.get_node_features(), dtype=torch.float)    # Node features
        x_2 = torch.tensor(scene_graph_3dssg.get_node_features(), dtype=torch.float)    # Node features

        # Get edge_index_1 and edge_index_2
        sources_1, targets_1, features_1 = scene_graph_human.get_edge_s_t_feats()
        edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
        edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

        sources_2, targets_2, features_2 = scene_graph_3dssg.get_edge_s_t_feats()
        edge_index_2 = torch.tensor([sources_2, targets_2], dtype=torch.long)
        edge_attr_2 = torch.tensor(features_2, dtype=torch.float)

        # Mask node
        x_1_masked = mask_node(x_1)
        x_2_masked = mask_node(x_2)

        # Get Place Node Index
        _, place_node_1_idx = scene_graph_human.get_place_node_idx()
        _, place_node_2_idx = scene_graph_3dssg.get_place_node_idx()

        # Make Cross Graph
        edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1_masked.shape, x_2_masked.shape)
        edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2_masked.shape, x_1_masked.shape)

        # Reset some values in the model
        model.edge_index_1_cross, model.edge_attr_1_cross = edge_index_1_cross, edge_attr_1_cross
        model.edge_index_2_cross, model.edge_attr_2_cross = edge_index_2_cross, edge_attr_2_cross

        # Get model output
        out1, out2, out_matching = model(x_1_masked, x_2_masked, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2, place_node_1_idx, place_node_2_idx)

        # print("out1", out1)
        # print("out2", out2)
        print("out_matching probability of test graph", out_matching)

if __name__ == '__main__':
    # Load scene graph
    scene_graph_human = SceneGraph('human+GPT', '0958224e-e2c2-2de1-943b-38e36345e2e7', raw_json='../output_clean/0958224e-e2c2-2de1-943b-38e36345e2e7/2_300ms/0_gpt_clean.json')
    scene_graph_3dssg = SceneGraph('3DSSG', '0958224e-e2c2-2de1-943b-38e36345e2e7', euc_dist_thres=1.0)

    # Process graph such that there are no gaps in indices and all nodes index from 0
    scene_graph_human.to_pyg()
    scene_graph_3dssg.to_pyg()

    # Make Hierarchical node that has an edge connecting to all other nodes
    scene_graph_human.add_place_node()
    scene_graph_3dssg.add_place_node()

    model = train_dummy_big_gnn(scene_graph_3dssg, scene_graph_3dssg)

    # Evaluate
    scene_graph_human_eval = SceneGraph('human+GPT', '0cac75ce-8d6f-2d13-8cf1-add4e795b9b0', raw_json='../output_clean/0cac75ce-8d6f-2d13-8cf1-add4e795b9b0/1_300ms/0_gpt_clean.json')
    scene_graph_3dssg_eval = SceneGraph('3DSSG', '0958224e-e2c2-2de1-943b-38e36345e2e7', euc_dist_thres=1.0)
    evaluate_model(model, scene_graph_human_eval, scene_graph_3dssg_eval)

    # TODO: Train on a set of 10 graphs
    # TODO: Design validation metric
    #     - See if we can recover the masked nodes to a good degree
    #     - Check the vector distance between the masked nodes
    #     - Check if we recover the "word" that was masked
    # TODO: Validate on 10 examples to see if we learn anything
    # TODO: Refactor everything and make it clean to swap out with GATConv or something else