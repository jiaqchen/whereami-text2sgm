###################################### DATA ######################################

import copy
import wandb
import random
import argparse
import os
import tqdm
import traceback
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, TransformerConv

from sg_dataloader import SceneGraph
from hungarian.sinkhorn import get_optimal_transport_scores, optimal_transport_list, get_subgraph
from utils import print_closest_words, make_cross_graph, mask_node, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

###################################### MODEL ######################################

class BigGNN(nn.Module):
    # NOTE: The "place node" needs to be used during training, in the forward pass, 
    # and also during evaluation?, and needs to be different for every graph pairing

    def __init__(self):
        super().__init__()
        self.N = 1 # Number of attention layers, all same sizes now, TODO: need to try different sizes
        self.TextSelfAttentionLayers = nn.ModuleList()
        self.GraphSelfAttentionLayers = nn.ModuleList()
        self.TextCrossAttentionLayers = nn.ModuleList()
        self.GraphCrossAttentionLayers = nn.ModuleList()
        for _ in range(self.N):
            self.TextSelfAttentionLayers.append(SimpleGAT(300, 300, 300))
            self.GraphSelfAttentionLayers.append(SimpleGAT(300, 300, 300))
            self.TextCrossAttentionLayers.append(SimpleGAT(300, 300, 300))
            self.GraphCrossAttentionLayers.append(SimpleGAT(300, 300, 300))

        # MLP for predicting matching score between 0 and 1
        self.SceneText_MLP = nn.Sequential(
            nn.Linear(600, 600), # TODO: input dimension is hardcoded now
            nn.ReLU(),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2_pos, 
                edge_index_1, edge_index_2_pos, 
                edge_attr_1, edge_attr_2_pos, 
                place_node_1_idx=None, place_node_2_idx_pos=None):

        for i in range(self.N):
            textSelfAttention = self.TextSelfAttentionLayers[i]
            graphSelfAttention = self.GraphSelfAttentionLayers[i]
            textCrossAttention = self.TextCrossAttentionLayers[i]
            graphCrossAttention = self.GraphCrossAttentionLayers[i]

            ############# Self Attention #############
            
            x_1 = textSelfAttention(x_1, edge_index_1, edge_attr_1)
            x_2_pos = graphSelfAttention(x_2_pos, edge_index_2_pos, edge_attr_2_pos)

            # Length of x_1 and x_2_pos
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2_pos.shape[0]

            ############# Cross Attention #############

            # Make Cross Attention Graphs
            edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1.shape, x_2_pos.shape) # First half of x_1_cross should be the original x_1
            edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2_pos.shape, x_1.shape) # First half of x_2_cross should be the original x_2_pos

            # Concatenate x_1 and x_2_pos
            x_1_cross = torch.cat((x_1, x_2_pos), dim=0)
            x_2_cross = torch.cat((x_2_pos, x_1), dim=0)

            # Cross Attention
            x_1_cross = textCrossAttention(x_1_cross, edge_index_1_cross, edge_attr_1_cross)
            x_2_cross = graphCrossAttention(x_2_cross, edge_index_2_cross, edge_attr_2_cross)

            # Get the first len_x_1 nodes from x_1_cross
            x_1 = x_1_cross[:len_x_1] # TODO: Oh, this could be weird....... need to make sure the nodes and indices line up here
            x_2_pos = x_2_cross[:len_x_2]

            # Batch Norm, am I using the Batch Norm correctly?
            x_1 = F.normalize(x_1, p=2, dim=1)
            x_2_pos = F.normalize(x_2_pos, p=2, dim=1)
        
        # Global average pooling
        x_1_pooled = torch.mean(x_1, dim=0)
        x_2_pos_pooled = torch.mean(x_2_pos, dim=0)

        return x_1_pooled, x_2_pos_pooled

class SimpleGAT(MessagePassing):
    # Simple one layer GATConv
    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super(SimpleGAT, self).__init__(aggr='add')  # "add" aggregation
        self.TransformerConv_nodes = TransformerConv(in_channels_node, out_channels, heads=8, concat=False, dropout=0.2)
        # self.GATConv_nodes = GATConv(in_channels_node, out_channels, heads=1, dropout=0.7)
        # self.GCNConv_nodes = GCNConv(in_channels_node, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.TransformerConv_nodes(x, edge_index)
        # x = self.GATConv_nodes(x, edge_index)

        x = F.relu(x)
        # Dropout
        # x = F.dropout(x, p=0.5, training=self.training)
        return x

# class SimpleGCN(MessagePassing):
#     # Simple one layer GATConv
#     def __init__(self, in_channels_node, in_channels_edge, out_channels):
#         super(SimpleGCN, self).__init__(aggr='add')  # "add" aggregation
#         # self.TransformerConv_nodes = TransformerConv(in_channels_node, out_channels, heads=1, dropout=0.8)
#         self.GCNConv_nodes = GCNConv(in_channels_node, out_channels)

#     def forward(self, x, edge_index, edge_attr):
#         x = self.GCNConv_nodes(x, edge_index)
#         # x = self.TransformerConv_nodes(x, edge_index)

#         # Take average over heads (TODO: for multi-head attention, currently not working)
#         # x = x.view(x.size(0), -1, self.GATConv_nodes.heads)  # Shape: [num_nodes, out_channels, num_heads]
#         # x = torch.mean(x, dim=-1)

#         # x = F.relu(x)
#         # Dropout
#         x = F.dropout(x, p=0.5, training=self.training)
    
#         # Normalize
#         x = F.normalize(x, p=2, dim=1)
#         return x
    
###################################### TRAIN ######################################

def train_dummy_big_gnn(list_of_graph1, list_of_graph2_dict):
    # Define model
    model = BigGNN() # TODO: input output channels are hardcoded now, need to figure that out
    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # assert(len(list_of_graph1) == len(list_of_graph2)) # TODO: not true anymore
    batch_size = args.batch_size
    for epoch in range(args.epoch):
        # for graph1, graph2_pos in zip(list_of_graph1, list_of_graph2):
        batch_hard_coded = 0
        for graph1 in list_of_graph1:
            loss = 0
            graph2_pos = list_of_graph2_dict[graph1.scene_id]
            # Turn graph2_pos, and graph2_neg into subgraphs that Hungarian match the nodes in graph1
            graph2_keys = list(list_of_graph2_dict.keys())
            graph2_keys.remove(graph1.scene_id)
            graph2_neg = list_of_graph2_dict[random.choice(graph2_keys)] # Negative example

            output_pos = optimal_transport_list(graph1, graph2_pos)
            output_neg = optimal_transport_list(graph1, graph2_neg)

            _, graph2_pos = get_subgraph(output_pos, graph1, graph2_pos, args.eps) # for now graph1 doesn't change
            _, graph2_neg = get_subgraph(output_neg, graph1, graph2_neg, args.eps) 

            # Nodes
            x_1 = torch.tensor(graph1.get_node_features(), dtype=torch.float)            # Node features
            x_2_pos = torch.tensor(graph2_pos.get_node_features(), dtype=torch.float)    # Node features
            x_2_neg = torch.tensor(graph2_neg.get_node_features(), dtype=torch.float)    # Node features

            min_nodes = 3
            if x_1.shape[0] <= min_nodes or x_2_pos.shape[0] <= min_nodes or x_2_neg.shape[0] <= min_nodes:
                continue

            # Edges
            sources_1, targets_1, features_1 = graph1.get_edge_s_t_feats()
            assert(len(sources_1) == len(targets_1) == len(features_1))
            edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
            edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

            source_2_pos, targets_2_pos, features_2_pos = graph2_pos.get_edge_s_t_feats()
            assert(len(source_2_pos) == len(targets_2_pos) == len(features_2_pos))
            edge_index_2_pos = torch.tensor([source_2_pos, targets_2_pos], dtype=torch.long)
            edge_attr_2_pos = torch.tensor(features_2_pos, dtype=torch.float)

            source_2_neg, targets_2_neg, features_2_neg = graph2_neg.get_edge_s_t_feats()
            assert(len(source_2_neg) == len(targets_2_neg) == len(features_2_neg))
            edge_index_2_neg = torch.tensor([source_2_neg, targets_2_neg], dtype=torch.long)
            edge_attr_2_neg = torch.tensor(features_2_neg, dtype=torch.float)

            # Get Place Node Index
            _, place_node_1_idx = graph1.get_place_node_idx()
            _, place_node_2_idx_pos = graph2_pos.get_place_node_idx()
            _, place_node_2_idx_neg = graph2_neg.get_place_node_idx()

            # Mask node
            # x_1_masked, _ = mask_node(x_1, p=0.2)
            # x_2_masked, _ = mask_node(x_2_pos, p=0.2)

            # Normalize input
            # x_1_masked = F.normalize(x_1_masked, p=2, dim=1)
            # x_2_masked = F.normalize(x_2_masked, p=2, dim=1)

            # TRAINING STEP
            # Go through all data in one epoch
            # if (batch_hard_coded % batch_size == 0):
            optimizer.zero_grad() # Clear gradients. # Must call before loss.backward() to avoid accumulating gradients from previous batches

            # TODO: OCT 9 2023 The input should just be a graph pair, with the pos and neg encoded within the loss function
            x_1_pos, x_2_pos = model(x_1, x_2_pos, 
                                            edge_index_1, edge_index_2_pos, 
                                            edge_attr_1, edge_attr_2_pos,
                                            place_node_1_idx, place_node_2_idx_pos) # Perform a single forward pass.
            x_1_neg, x_2_neg = model(x_1, x_2_neg,
                                            edge_index_1, edge_index_2_neg,
                                            edge_attr_1, edge_attr_2_neg,
                                            place_node_1_idx, place_node_2_idx_neg) # Perform a single forward pass.

            # Cosine distance
            loss1 = 1 - F.cosine_similarity(x_1_pos, x_2_pos, dim=0) # Compute the loss. force to 0
            loss2 = 1 - F.cosine_similarity(x_1_neg, x_2_neg, dim=0) # Compute the loss. force to 2

            # MSE loss with cosine similarity
            # loss1 = F.mse_loss(loss1, torch.tensor([1.0], dtype=torch.float)) # Compute the loss.
            # loss2 = F.mse_loss(loss2, torch.tensor([1.0], dtype=torch.float)) # Compute the loss.

            # loss = ((1 - torch.cat((loss1, loss2), dim=0)).sum()) + loss3
            loss += loss1.sum() - loss2.sum() + 2

            if (batch_hard_coded % batch_size == 0):
                # print first 10 values of the outputs x_1_pos, x_2_pos, x_1_neg, x_2_neg

                loss.backward() # Derive gradients.
                optimizer.step() # Update parameters based on gradients.

                wandb.log({"loss_per_batch": loss.item()})
                loss = 0
                batch_hard_coded = 0
            else:
                batch_hard_coded += 1

        # Print loss
        if epoch % 10 == 0:
            wandb.log({"loss_per_epoch": loss.item()})

        # # Check accuracy
        # if epoch % 5 == 0 and epoch != 0:
        #     out1_vector = out1.detach().numpy()
        #     out2_vector = out2.detach().numpy()
        #     x_1_vector = x_1.detach().numpy()
        #     x_2_vector = x_2_pos.detach().numpy()
        #     print_closest_words(out1_vector, x_1_vector, first_n=x_1_vector.shape[0])
        #     print()
        #     print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])

        # Check accuracy of classification
        if epoch % 10 == 0:
            acc = evaluate_classification(model)
            wandb.log({"accuracy": acc})

    return model

def evaluate_nodes(model):
    # Go through all human scene graphs and get a score for each.
    scene_ids = os.listdir('../output_clean')
    avg_accs = []
    avg_acc1s = []
    avg_acc2s = []
    i = 0
    for s in scene_ids:
        # raw json is the file inside the folder
        raw_json = os.listdir('../output_clean/' + s)[0]
        raw_json = '../output_clean/' + s + '/' + raw_json + '/0_gpt_clean.json'
        try:
            scene_graph_human_eval = SceneGraph('human+GPT', s, raw_json=raw_json)
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            print("Error with graph generation with human ", s)
            continue
        try:
            scene_graph_3dssg_eval = SceneGraph('3DSSG', s, euc_dist_thres=1.0)
        except Exception as e:
            # print(e)
            print("Error with graph generation with 3DSSG ", s)
            continue

        try:
            acc1, acc2, avg_acc, out1_vector, out2_vector, x_1_vector, x_2_vector = evaluate_model(model, scene_graph_human_eval, scene_graph_3dssg_eval)
        except Exception as e:
            # print(e)
            print(traceback.print_exc())
            print("Error with scene evaluation ", s)
            continue
        avg_accs.append(avg_acc)
        avg_acc1s.append(acc1)
        avg_acc2s.append(acc2)

        # every 5 iterations, show closest
        if i % 5 == 0:
            print("closest words for x_1")
            print_closest_words(out1_vector, x_1_vector, first_n=x_1_vector.shape[0])
            print("closest words for x_2_pos")
            print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])
        
        i += 1

    # Print overall results
    print("Average accuracy weighted by number of nodes masked: ", sum(avg_accs) / len(avg_accs))
    print("Average accuracy weighted by number of nodes masked for x_1 (from ScanScribe): ", sum(avg_acc1s) / len(avg_acc1s))
    print("Average accuracy weighted by number of nodes masked for x_2_pos (from 3DSSG): ", sum(avg_acc2s) / len(avg_acc2s))

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

        # Get x_1 and x_2_pos
        x_1 = torch.tensor(scene_graph_human.get_node_features(), dtype=torch.float)    # Node features
        x_2_pos = torch.tensor(scene_graph_3dssg.get_node_features(), dtype=torch.float)    # Node features

        # Get edge_index_1 and edge_index_2_pos
        sources_1, targets_1, features_1 = scene_graph_human.get_edge_s_t_feats()
        edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
        edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

        source_2_pos, targets_2_pos, features_2_pos = scene_graph_3dssg.get_edge_s_t_feats()
        edge_index_2_pos = torch.tensor([source_2_pos, targets_2_pos], dtype=torch.long)
        edge_attr_2_pos = torch.tensor(features_2_pos, dtype=torch.float)

        # Mask node
        x_1_masked, x_1_masked_rows = mask_node(x_1, p=0.1)
        x_2_masked, x_2_masked_rows = mask_node(x_2_pos, p=0.1)

        # Get Place Node Index
        _, place_node_1_idx = scene_graph_human.get_place_node_idx()
        _, place_node_2_idx_pos = scene_graph_3dssg.get_place_node_idx()

        # Make Cross Graph
        edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1_masked.shape, x_2_masked.shape)
        edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2_masked.shape, x_1_masked.shape)

        # Reset some values in the model
        model.edge_index_1_cross, model.edge_attr_1_cross = edge_index_1_cross, edge_attr_1_cross
        model.edge_index_2_cross, model.edge_attr_2_cross = edge_index_2_cross, edge_attr_2_cross

        # Get model output
        out1, out2, out_matching = model(x_1_masked, x_2_masked, edge_index_1, edge_index_2_pos, edge_attr_1, edge_attr_2_pos,
                                         place_node_1_idx, place_node_2_idx_pos)

        out1_vector = out1.detach().numpy()
        out2_vector = out2.detach().numpy()
        x_1_vector = x_1.detach().numpy()
        x_2_vector = x_2_pos.detach().numpy()
        # Use the masks
        out1_vector = out1_vector[x_1_masked_rows] # TODO: I think this means we are only calculating accuracy on the masked nodes, so that's good right?
        # print("out1_vector shape", out1_vector.shape)
        out2_vector = out2_vector[x_2_masked_rows]
        x_1_vector = x_1_vector[x_1_masked_rows]
        x_2_vector = x_2_vector[x_2_masked_rows]

        # print("closest words for x_1")
        # print_closest_words(out1_vector, x_1_vector, first_n=x_1_vector.shape[0])
        # print("closest words for x_2_pos")
        # print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])

        # Print Accuracy of just the masked nodes
        acc1 = accuracy_score(out1_vector, x_1_vector)
        acc2 = accuracy_score(out2_vector, x_2_vector)
        # print("Accuracy for x_1: ", acc1)
        # print("Accuracy for x_2_pos: ", acc2)
        # print("Average accuracy weighted by number of nodes masked: ", (acc1 * x_1_vector.shape[0] + acc2 * x_2_vector.shape[0]) / (x_1_vector.shape[0] + x_2_vector.shape[0]))
        avg_acc = (acc1 * x_1_vector.shape[0] + acc2 * x_2_vector.shape[0]) / (x_1_vector.shape[0] + x_2_vector.shape[0])
        return acc1, acc2, avg_acc, out1_vector, out2_vector, x_1_vector, x_2_vector

def evaluate_classification(model):
# Go through all human scene graphs and get a score for each.
    scene_ids = os.listdir('../output_clean')
    accuracies = []
    i = 0
    for s in scene_ids:
        # raw json is the file inside the folder
        raw_json = os.listdir('../output_clean/' + s)[0]
        raw_json = '../output_clean/' + s + '/' + raw_json + '/0_gpt_clean.json'
        try:
            scene_graph_human_eval = SceneGraph('human+GPT', s, raw_json=raw_json)
        except Exception as e:
            continue
        try:
            scene_graph_3dssg_eval = SceneGraph('3DSSG', s, euc_dist_thres=1.0)
        except Exception as e:
            continue
        try:
            # random s in scene_id
            s_rand = scene_ids.copy()
            s_rand.remove(s)
            s_rand = random.choice(s_rand)
            scene_graph_3dssg_eval_neg = SceneGraph('3DSSG', s_rand, euc_dist_thres=1.0)
        except:
            continue
        try:
            acc = evaluate_model_classification(model, scene_graph_human_eval, scene_graph_3dssg_eval, 1)
            acc_neg = evaluate_model_classification(model, scene_graph_human_eval, scene_graph_3dssg_eval_neg, 0)
        except Exception as e:
            continue
        accuracies.append(acc)
        accuracies.append(acc_neg)
        i += 1

    return sum(accuracies) / len(accuracies)

def evaluate_model_classification(model, scene_graph_human, scene_graph_3dssg, label):
    # Evaluate model on another graph pair
    model.eval()
    with torch.no_grad():
        # Process graph such that there are no gaps in indices and all nodes index from 0
        scene_graph_human.to_pyg()
        scene_graph_3dssg.to_pyg()

        # Make Hierarchical node that has an edge connecting to all other nodes
        scene_graph_human.add_place_node() # TODO: adding a place node is find for human+GPT graph I think
        scene_graph_3dssg.add_place_node() # TODO: we can reuse the 3dssg graphs, so need to reuse those 
                                           #       here instead of making new and adding place node

        # Get x_1 and x_2_pos
        x_1 = torch.tensor(scene_graph_human.get_node_features(), dtype=torch.float)    # Node features
        x_2_pos = torch.tensor(scene_graph_3dssg.get_node_features(), dtype=torch.float)    # Node features

        # Get edge_index_1 and edge_index_2_pos
        sources_1, targets_1, features_1 = scene_graph_human.get_edge_s_t_feats()
        edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
        edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

        source_2_pos, targets_2_pos, features_2_pos = scene_graph_3dssg.get_edge_s_t_feats()
        edge_index_2_pos = torch.tensor([source_2_pos, targets_2_pos], dtype=torch.long)
        edge_attr_2_pos = torch.tensor(features_2_pos, dtype=torch.float)

        # Mask node
        # x_1_masked, x_1_masked_rows = mask_node(x_1, p=0.1)
        # x_2_masked, x_2_masked_rows = mask_node(x_2_pos, p=0.1)

        # Get Place Node Index
        _, place_node_1_idx = scene_graph_human.get_place_node_idx()
        _, place_node_2_idx_pos = scene_graph_3dssg.get_place_node_idx()

        # Make Cross Graph
        # edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1_masked.shape, x_2_masked.shape)
        # edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2_masked.shape, x_1_masked.shape)

        # Reset some values in the model
        # model.edge_index_1_cross, model.edge_attr_1_cross = edge_index_1_cross, edge_attr_1_cross
        # model.edge_index_2_cross, model.edge_attr_2_cross = edge_index_2_cross, edge_attr_2_cross

        # Get model output
        _, _, out_matching = model(x_1, x_2_pos, edge_index_1, edge_index_2_pos, edge_attr_1, edge_attr_2_pos,
                                         place_node_1_idx, place_node_2_idx_pos)
        if label == 1:
            return out_matching > 0.5
        else:
            return out_matching < 0.5
    
if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--text_source', type=str, default='human+GPT', help='human+GPT or ScanScribe3DSSG+GPT') # ScanScribe3DSSG+GPT is GPT annotated from SG, and then reparsed back into a JSON lawl
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Hungarian matching')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    args = parser.parse_args()
    assert(args.text_source == 'human+GPT' or args.text_source == 'ScanScribe3DSSG+GPT')

    list_of_graph_3dssg_dict_room_label = None
    list_of_graph_text = None

    # We must have a list_of_graph_3dssg_dict_room_label
    if os.path.exists('list_of_graph_3dssg_dict_room_label.pt'):
        print("Using 3DSSG presaved scene graphs")
        list_of_graph_3dssg_dict_room_label = torch.load('list_of_graph_3dssg_dict_room_label.pt') 
    else: # Load 3DSSG graphs as dict
        scene_ids_3dssg = os.listdir('../../data/3DSSG/3RScan')
        list_of_graph_3dssg_dict_room_label = {}
        for scene_id in tqdm.tqdm(scene_ids_3dssg):
            try:
                scene_graph_3dssg = SceneGraph('3DSSG', scene_id, euc_dist_thres=1.0)
            except Exception as e:
                print("Error with loading 3DSSG scene graph scene ", scene_id)
                continue
            try:
                scene_graph_3dssg.to_pyg()
            except:
                continue
            scene_graph_3dssg.add_place_node() 
            list_of_graph_3dssg_dict_room_label[scene_id] = scene_graph_3dssg
        torch.save(list_of_graph_3dssg_dict_room_label, 'list_of_graph_3dssg_dict_room_label.pt')

    # Now load either ScanScribe3DSSG+GPT or human+GPT for the text source
    # 3DSSG is the set of target graphs, we use either human or GPT annotations as the text graph
    if args.text_source == 'ScanScribe3DSSG+GPT':
        print("Using ScanScribe3DSSG+GPT as text source")
        scene_ids = os.listdir('../scripts/scanscribe_json_gpt')
        
        # TODO: Try adding attributes to the features and saving another graph checkpoint
        if os.path.exists('list_of_graph_scanscribe_gpt_room_label.pt'):
            print("Using ScanScribe presaved text source")
            list_of_graph_text = torch.load('list_of_graph_scanscribe_gpt_room_label.pt')
        else:
            # Go through folders
            list_of_graph_text = []
            for scene_id in scene_ids:
                # Get files in folder
                texts = os.listdir('../scripts/scanscribe_json_gpt/' + scene_id)
                for text_i in texts:
                    # Load scene graph
                    try:
                        scene_graph_scanscribe_gpt = SceneGraph('human+GPT', scene_id, raw_json='../scripts/scanscribe_json_gpt/' + scene_id + '/' + text_i) # ScanScribe3DSSG+GPT has the same JSON signature as human+GPT
                    except Exception as e:
                        # print(e)
                        # print(traceback.format_exc())
                        print("Error with loading ScanScribe3DSSG+GPT scene graph ", scene_id, "           text ", text_i)
                        continue

                    # Process graph such that there are no gaps in indices and all nodes index from 0
                    try:
                        scene_graph_scanscribe_gpt.to_pyg()
                    except Exception as e:
                        # print(e)
                        # print(traceback.format_exc())
                        print("Error with conversion to pyg graph for ScanScribe3DSSG+GPT scene graph ", scene_id, "           text ", text_i)
                        continue
                    scene_graph_scanscribe_gpt.add_place_node()
                    list_of_graph_text.append(scene_graph_scanscribe_gpt)
                
            # Save list to file to access later
            torch.save(list_of_graph_text, 'list_of_graph_scanscribe_gpt_room_label.pt')

    elif args.text_source == 'human+GPT':
        # Load Dataset (getting matched scene text pairs from human annotations)
        scene_ids = os.listdir('../output_clean')

        # Load list of graphs
        if os.path.exists('list_of_graph_human.pt'):
            print("Using presaved Human+GPT scene graph")
            list_of_graph_text = torch.load('list_of_graph_human.pt')
        else:
            # Go through folders
            list_of_graph_text = []
            for scene_id in scene_ids:
                print("Loading scene ", scene_id)
                # Load scene graph
                human_folder = '../output_clean/' + scene_id
                human_subfolder = os.listdir(human_folder)[0]
                try:
                    scene_graph_human = SceneGraph('human+GPT', scene_id, raw_json='../output_clean/' + scene_id + '/' + human_subfolder + '/0_gpt_clean.json')
                except Exception as e:
                    # print(e)
                    print("Error with creating human+GPT scene graph ", scene_id)
                    continue

                # Process graph such that there are no gaps in indices and all nodes index from 0
                scene_graph_human.to_pyg()
                scene_graph_human.add_place_node()
                list_of_graph_text.append(scene_graph_human)

            # Save list to file to access later
            torch.save(list_of_graph_text, 'list_of_graph_human.pt')

    if (list_of_graph_text is None or len(list_of_graph_3dssg_dict_room_label) == 0):
        print("Error loading data")
        exit()

    # Go through both and make sure none of them have a "place" node
    for graph in list_of_graph_3dssg_dict_room_label.values():
        for node in graph.get_nodes():
            if node.node_type == "place":
                # Remove the node
                graph.remove_node(node)
                break
        
    for graph in list_of_graph_text:
        for node in graph.get_nodes():
            if node.node_type == "place":
                # Remove the node
                graph.remove_node(node)
                break

    wandb.init(project="simplegnn",
            config={
                "architecture": "self attention cross attention",
                "dataset": "ScanScribe og", # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
                "epochs": args.epoch,
                "batch_size": args.batch_size,
            })

    model = train_dummy_big_gnn(list_of_graph_text, list_of_graph_3dssg_dict_room_label)

    # Evaluate
    final_accuracy = evaluate_classification(model)
    # evaluate_nodes(model)
    print("Final accuracy: ", final_accuracy)




























    # TODO: Train on a set of 10 graphs
    # TODO: Design validation metric
    #     - See if we can recover the masked nodes to a good degree
    #     - Check the vector distance between the masked nodes
    #     - Check if we recover the "word" that was masked
    # TODO: Validate on 10 examples to see if we learn anything
    # TODO: Refactor everything and make it clean to swap out with GATConv or something else
    # TODO: After refactoring, add multi-headed attention
    # TODO: Separate GCN to learn a "place" node to do comparions later

    # TODO: merge datapoints together
    # TODO: train using a contrastive loss (need to set up the dataset for this)

    # TODO: clean the data to remove "floor", "wall", etc. nodes (but I think I already do this?)
    # TODO: add relu between each transformer layer, in between self and cross attntion
    # TODO: print the accuracy alongside the loss
    # TODO: mask only certain nodes and make sure none overlap
    # TODO: change the loss to (binary) properly model what is supposed to be modeled?
    # TODO: why does Transformer work better than GATConv?
    # TODO: check that the way I calculate Loss is correct... especially with the batching
    # TODO: check the weights to see if it's actually learning something helpful??? [I'm not sure if they're changing that much?]
    # TODO: omg clean up the data my dude maybe that's why it's not working, and also I think transformers need a lot of data