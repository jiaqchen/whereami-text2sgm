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

import sys
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc')
sys.path.append('../') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models')
from playground.graph_models.data_processing.sg_dataloader import SceneGraph, Node
from hungarian.sinkhorn import get_optimal_transport_scores, optimal_transport_between_two_graphs, get_subgraph
from playground.graph_models.src.utils import print_closest_words, make_cross_graph, mask_node, accuracy_score, verify_subgraph

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

# python3 simplegnn.py --epoch 30 --text_source ScanScribe3DSSG+GPT --batch_size 4 --mode online --traintestsplit 0.9 --seed 0 --one_datapoint 16 --top_k 10 --out_of 100 --dbscann_eps 0.05 --sinkhorn_thr 0.1 --training_out_of 16 --training_top_k 1 --lr 0.001 --weight_decay 5e-4
###################################### MODEL ######################################

class BigGNN(nn.Module):
    # NOTE: The "place node" needs to be used during training, in the forward pass, 
    # and also during evaluation?, and needs to be different for every graph pairing

    def __init__(self):
        super().__init__()
        self.N = args.N # Number of attention layers, all same sizes now, TODO: need to try different sizes
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
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

    def forward(self, x_1, x_2_pos, 
                edge_index_1, edge_index_2_pos, 
                edge_attr_1, edge_attr_2_pos, 
                place_node_1_idx=None, place_node_2_idx_pos=None):
        
        # add everything to cuda
        x_1 = x_1.to('cuda')
        x_2_pos = x_2_pos.to('cuda')
        edge_index_1 = edge_index_1.to('cuda')
        edge_index_2_pos = edge_index_2_pos.to('cuda')
        edge_attr_1 = edge_attr_1.to('cuda')
        edge_attr_2_pos = edge_attr_2_pos.to('cuda')

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

            x_1_cross = torch.cat((x_1, x_2_pos), dim=0)
            x_2_cross = torch.cat((x_2_pos, x_1), dim=0)

            # Cross Attention
            x_1_cross = textCrossAttention(x_1_cross.to('cuda'), edge_index_1_cross.to('cuda'), edge_attr_1_cross.to('cuda'))
            x_2_cross = graphCrossAttention(x_2_cross.to('cuda'), edge_index_2_cross.to('cuda'), edge_attr_2_cross.to('cuda'))

            # Get the first len_x_1 nodes from x_1_cross
            x_1 = x_1_cross[:len_x_1] # TODO: Oh, this could be weird....... need to make sure the nodes and indices line up here
            x_2_pos = x_2_cross[:len_x_2]

            # Batch Norm, am I using the Batch Norm correctly?
            x_1 = F.normalize(x_1, p=2, dim=1)
            x_2_pos = F.normalize(x_2_pos, p=2, dim=1)
        
        # Global average pooling
        x_1_pooled = torch.mean(x_1, dim=0)
        x_2_pos_pooled = torch.mean(x_2_pos, dim=0)

        # Concatenate and feed into SceneTextMLP
        x_concat = torch.cat((x_1_pooled, x_2_pos_pooled), dim=0)
        out_matching = self.SceneText_MLP(x_concat)

        print("out_matching: ", out_matching)
        # print("x_1_pooled: ", x_1_pooled)
        # print("x_2_pos_pooled: ", x_2_pos_pooled)
        return x_1_pooled, x_2_pos_pooled, out_matching

class SimpleGAT(MessagePassing):
    # Simple one layer GATConv
    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super(SimpleGAT, self).__init__(aggr='add')  # "add" aggregation
        self.TransformerConv_nodes = TransformerConv(in_channels_node, out_channels, heads=4, concat=False, dropout=0.5)
        # self.GATConv_nodes = GATConv(in_channels_node, out_channels, heads=1, dropout=0.7)
        # self.GCNConv_nodes = GCNConv(in_channels_node, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.TransformerConv_nodes(x, edge_index)
        # x = self.GATConv_nodes(x, edge_index)

        # x = F.relu(x)
        # Dropout
        # x = F.dropout(x, p=0.5, training=self.training)
        return x
    
###################################### TRAIN ######################################

def train_dummy_big_gnn(list_of_graph1, list_of_graph2_dict, text_test, graph_test):
    # # first half of list_of_graph1 is the positive examples # SWAP BETWEEN SPLIT V NO SPLIT
    # list_of_graph1_pos = list_of_graph1[0:len(list_of_graph1)//2]
    # list_of_graph2_neg = []
    # for g in list_of_graph1[len(list_of_graph1)//2:]:
    #     list_of_graph2_neg.append(g.scene_id)
    text_test = text_test[0:args.training_out_of]
    graph_test = {t.scene_id: graph_test[t.scene_id] for t in text_test}
        
    # Define model
    model = BigGNN().to('cuda') # TODO: input output channels are hardcoded now, need to figure that out
    # Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # assert(len(list_of_graph1) == len(list_of_graph2)) # TODO: not true anymore
    batch_size = args.batch_size
    for epoch in range(args.epoch):
        # for graph1, graph2_pos in zip(list_of_graph1, list_of_graph2):
        batch_hard_coded = 0
        # for graph1 in list_of_graph1_pos: # SWAP BETWEEN SPLIT V NO SPLIT
        # shuffle list_of_graph1
        random.shuffle(list_of_graph1)
        for graph1 in list_of_graph1:
            loss = 0
            graph2_pos = list_of_graph2_dict[graph1.scene_id]
            # Turn graph2_pos, and graph2_neg into subgraphs that Hungarian match the nodes in graph1
            graph2_keys = list(list_of_graph2_dict.keys())
            graph2_keys.remove(graph1.scene_id)
            graph2_neg = list_of_graph2_dict[random.choice(graph2_keys)] # Negative example
            # graph2_neg = list_of_graph2_dict[random.choice(list_of_graph2_neg)] # Negative example # SWAP BETWEEN SPLIT V NO SPLIT

            output_pos = optimal_transport_between_two_graphs(graph1, graph2_pos, args.sinkhorn_thr)
            output_neg = optimal_transport_between_two_graphs(graph1, graph2_neg, args.sinkhorn_thr)

            new_graph2_pos = None
            new_graph2_neg = None
            if not all([x == -1 for x in output_pos['matches0'][0]]):
                _, new_graph2_pos, graph2_pos_clusters = get_subgraph(output_pos, graph1, graph2_pos, args.dbscann_eps)
            
            if not all([x == -1 for x in output_neg['matches0'][0]]):
                _, new_graph2_neg, graph2_neg_clusters = get_subgraph(output_neg, graph1, graph2_neg, args.dbscann_eps)

            # Verify subgraph that it's correct
            if (new_graph2_pos is not None):
                # print("################ positive example ################\n")
                # verify_subgraph(graph1, new_graph2_pos, graph2_pos, output_pos, clusters=graph2_pos_clusters)
                graph2_pos = new_graph2_pos
            if (new_graph2_neg is not None):
                # print("################ negative example ################\n")
                # verify_subgraph(graph1, new_graph2_neg, graph2_neg, output_neg, clusters=graph2_neg_clusters)
                graph2_neg = new_graph2_neg

            # Nodes
            x_1 = torch.tensor(graph1.get_node_features(), dtype=torch.float).to('cuda')            # Node features
            x_2_pos = torch.tensor(graph2_pos.get_node_features(), dtype=torch.float).to('cuda')    # Node features
            x_2_neg = torch.tensor(graph2_neg.get_node_features(), dtype=torch.float).to('cuda')    # Node features

            # min_nodes = 3
            # if x_1.shape[0] <= min_nodes or x_2_pos.shape[0] <= min_nodes or x_2_neg.shape[0] <= min_nodes:
            #     continue

            # Edges
            sources_1, targets_1, features_1 = graph1.get_edge_s_t_feats()
            assert(len(sources_1) == len(targets_1) == len(features_1))
            edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long).to('cuda')
            edge_attr_1 = torch.tensor(features_1, dtype=torch.float).to('cuda')

            source_2_pos, targets_2_pos, features_2_pos = graph2_pos.get_edge_s_t_feats()
            assert(len(source_2_pos) == len(targets_2_pos) == len(features_2_pos))
            edge_index_2_pos = torch.tensor([source_2_pos, targets_2_pos], dtype=torch.long).to('cuda')
            edge_attr_2_pos = torch.tensor(features_2_pos, dtype=torch.float).to('cuda')

            source_2_neg, targets_2_neg, features_2_neg = graph2_neg.get_edge_s_t_feats()
            assert(len(source_2_neg) == len(targets_2_neg) == len(features_2_neg))
            edge_index_2_neg = torch.tensor([source_2_neg, targets_2_neg], dtype=torch.long).to('cuda')
            edge_attr_2_neg = torch.tensor(features_2_neg, dtype=torch.float).to('cuda')

            # Get Place Node Index
            _, place_node_1_idx = graph1.get_place_node_idx()
            _, place_node_2_idx_pos = graph2_pos.get_place_node_idx()
            _, place_node_2_idx_neg = graph2_neg.get_place_node_idx()

            # TRAINING STEP
            # Go through all data in one epoch
            if (batch_hard_coded % batch_size == 0):
                optimizer.zero_grad() # Clear gradients. # Must call before loss.backward() to avoid accumulating gradients from previous batches

            # TODO: OCT 9 2023 The input should just be a graph pair, with the pos and neg encoded within the loss function
            x_1_pos, x_2_pos, match_prob_pos = model(x_1.to('cuda'), x_2_pos.to('cuda'), 
                                            edge_index_1.to('cuda'), edge_index_2_pos.to('cuda'), 
                                            edge_attr_1.to('cuda'), edge_attr_2_pos.to('cuda'))
                                            # place_node_1_idx, place_node_2_idx_pos) # Perform a single forward pass.
            x_1_neg, x_2_neg, match_prob_neg = model(x_1.to('cuda'), x_2_neg.to('cuda'),
                                            edge_index_1.to('cuda'), edge_index_2_neg.to('cuda'),
                                            edge_attr_1.to('cuda'), edge_attr_2_neg.to('cuda'))
                                            # place_node_1_idx, place_node_2_idx_neg) # Perform a single forward pass.

            # Normalize
            x_1_pos = F.normalize(x_1_pos, p=2, dim=0)
            x_2_pos = F.normalize(x_2_pos, p=2, dim=0)
            x_1_neg = F.normalize(x_1_neg, p=2, dim=0)
            x_2_neg = F.normalize(x_2_neg, p=2, dim=0)

            # Cosine distance
            loss1 = 1 - F.cosine_similarity(x_1_pos, x_2_pos, dim=0) # [0, 2] 0 is good
            loss2 = 1 - F.cosine_similarity(x_1_neg, x_2_neg, dim=0) # [0, 2] 2 is good
            bias = 5
            # loss1 = 1 - torch.sigmoid(bias*loss1) #[0,1] 0 is good
            # loss2 = torch.sigmoid(bias*loss2) #[0,1] 0 is good
            # make sure x_1_pos and x_2_neg are as similar as possible
            loss3 = F.mse_loss(x_1_pos, x_2_neg)
            loss4 = (1 - match_prob_pos) + match_prob_neg

            loss += loss1.sum() + 2 - loss2.sum() + loss4# + loss3.sum()

            if (batch_hard_coded % batch_size == 0):
                wandb.log({"loss1": loss1.sum().item(),
                            "loss2": loss2.sum().item(),
                            "loss3": loss3.sum().item(),
                            "match_prob_pos": match_prob_pos.item(),
                            "match_prob_neg": match_prob_neg.item()})
                loss = loss / batch_size
                loss.backward() # Derive gradients.
                optimizer.step() # Update parameters based on gradients.
                wandb.log({"loss_per_batch": loss.item()})
                epoch_loss = loss
                loss = 0
                batch_hard_coded = 0
            else:
                batch_hard_coded += 1

        # Print loss
        if epoch % 1 == 0:
            wandb.log({"loss_per_epoch": epoch_loss.item()})

        # Check accuracy of classification
        if epoch % 1 == 0:
            acc, _ = evaluate_model(model, list_of_graph1, list_of_graph2_dict, out_of=args.training_out_of, top_k=args.training_top_k)
            acc_test, _ = evaluate_model(model, text_test, graph_test, out_of=args.training_out_of, top_k=args.training_top_k)
            wandb.log({"accuracy_per_epochs": acc,
                "accuracy_test_per_epoch": acc_test})


    return model

def evaluate_model(model, list_of_graph1, list_of_graph2_dict, out_of=100, top_k=10):
    model.eval()
    accuracies = []
    for graph1 in list_of_graph1:
        graph2_pos = list_of_graph2_dict[graph1.scene_id]
        # Turn graph2_pos, and graph2_neg into subgraphs that Hungarian match the nodes in graph1
        graph2_keys = list(list_of_graph2_dict.keys())
        graph2_keys.remove(graph1.scene_id)
        samples = random.sample(graph2_keys, out_of-1)
        graph2_negs = [list_of_graph2_dict[s] for s in samples] # Negative example
        graph2_negs.append(graph2_pos) # +1 more sample
        random.shuffle(graph2_negs)
        matching_probs = []
        for graph2_neg in graph2_negs:
            output_neg = optimal_transport_between_two_graphs(graph1, graph2_neg, args.sinkhorn_thr)
            
            if not all([x == -1 for x in output_neg['matches0'][0]]):
                _, graph2_neg, graph2_neg_clusters = get_subgraph(output_neg, graph1, graph2_neg, args.dbscann_eps)

            # Nodes
            x_1 = torch.tensor(graph1.get_node_features(), dtype=torch.float)            # Node features
            x_2_neg = torch.tensor(graph2_neg.get_node_features(), dtype=torch.float)    # Node features

            # Edges
            sources_1, targets_1, features_1 = graph1.get_edge_s_t_feats()
            assert(len(sources_1) == len(targets_1) == len(features_1))
            edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
            edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

            source_2_neg, targets_2_neg, features_2_neg = graph2_neg.get_edge_s_t_feats()
            assert(len(source_2_neg) == len(targets_2_neg) == len(features_2_neg))
            edge_index_2_neg = torch.tensor([source_2_neg, targets_2_neg], dtype=torch.long)
            edge_attr_2_neg = torch.tensor(features_2_neg, dtype=torch.float)

            # Get Place Node Index
            _, place_node_1_idx = graph1.get_place_node_idx()
            _, place_node_2_idx_neg = graph2_neg.get_place_node_idx()

            x_1_neg, x_2_neg, match_prob_neg = model(x_1, x_2_neg,
                                            edge_index_1, edge_index_2_neg,
                                            edge_attr_1, edge_attr_2_neg,
                                            place_node_1_idx, place_node_2_idx_neg)
            # Cosine distance
            loss2 = 1 - F.cosine_similarity(x_1_neg, x_2_neg, dim=0) # Compute the loss. force to 2
            matching_probs.append(match_prob_neg.item())

        # Get top k
        top_k_idx = sorted(range(len(matching_probs)), key=lambda i: matching_probs[i])[-top_k:]

        # Get top k scene_ids from graph2_negs
        top_k_scene_ids = []
        for idx in top_k_idx:
            top_k_scene_ids.append(graph2_negs[idx].scene_id)

        # Get accuracy
        if graph1.scene_id in top_k_scene_ids:
            accuracies.append(1)
        else:
            accuracies.append(0)

    return sum(accuracies) / len(accuracies), len(accuracies)
    
def train_test_split(list_of_text_graph, test_size):
    random.shuffle(list_of_text_graph)
    test_size = int(len(list_of_text_graph) * test_size)
    list_of_text_graph_test = list_of_text_graph[:test_size]
    list_of_text_graph_train = list_of_text_graph[test_size:]
    return list_of_text_graph_train, list_of_text_graph_test

def split_3dssg(list_text_graph_train, list_text_graph_test, dict_3dssg_graph):
    dict_3dssg_graph_train = {}
    dict_3dssg_graph_test = {}
    for text_graph in list_text_graph_train:
        dict_3dssg_graph_train[text_graph.scene_id] = dict_3dssg_graph[text_graph.scene_id]
    for text_graph in list_text_graph_test:
        dict_3dssg_graph_test[text_graph.scene_id] = dict_3dssg_graph[text_graph.scene_id]
    return dict_3dssg_graph_train, dict_3dssg_graph_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--text_source', type=str, default='human+GPT', help='human+GPT or ScanScribe3DSSG+GPT') # ScanScribe3DSSG+GPT is GPT annotated from SG, and then reparsed back into a JSON lawl
    parser.add_argument('--dbscann_eps', type=float, default=0.05, help='epsilon for Hungarian matching')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--mode', type=str, default='online', help='online or offline or disabled')
    parser.add_argument('--traintestsplit', type=float, default=0.5, help='train test split for the dataset')
    parser.add_argument('--seed', type=int, default=0, help='seed for random')
    parser.add_argument('--one_datapoint', type=int, default=None)
    parser.add_argument('--top_k', type=int, default=10, help='top k accuracy')
    parser.add_argument('--out_of', type=int, default=20, help='out of how many to test accuracy')
    parser.add_argument('--sinkhorn_thr', type=float, default=0.05, help='sinkhorn threshold')
    parser.add_argument('--training_out_of', type=int, default=None, help='out of how many to test accuracy')
    parser.add_argument('--training_top_k', type=int, default=None, help='top k accuracy')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay')
    parser.add_argument('--N', type=int, default=1, help='number of attention layers')
    args = parser.parse_args()
    random.seed(args.seed)
    assert(args.text_source == 'human+GPT' or args.text_source == 'ScanScribe3DSSG+GPT')

    list_of_graph_3dssg_dict_room_label = None
    list_of_graph_text = None

    # We must have a list_of_graph_3dssg_dict_room_label
    if os.path.exists('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/list_of_graph_3dssg_dict_label_vec_no_attrib_no_room_node.pt'):
        print("Using 3DSSG presaved scene graphs")
        list_of_graph_3dssg_dict_room_label = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/list_of_graph_3dssg_dict_label_vec_no_attrib_no_room_node.pt') 
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
    
        # TODO: Try adding attributes to the features and saving another graph checkpoint
        if os.path.exists('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/list_of_graph_text_label_vec_no_attrib_no_room_node.pt'):
            print("Using ScanScribe presaved text source")
            list_of_graph_text = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/list_of_graph_text_label_vec_no_attrib_no_room_node.pt')
        else:
            scene_ids = os.listdir('../scripts/scanscribe_json_gpt')
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

    ############################################################ Removing place node and features are just the label vector, and then saving
    # # Go through both and make sure none of them have a "place" node
    # for graph in list_of_graph_3dssg_dict_room_label.values():
    #     for node in graph.get_nodes():
    #         # redo the feature to make sure that it doesn't have the attributes included
    #         node.features = node.set_features_without_attributes(node.label)
    #         if node.node_type == "place":
    #             # Remove the node
    #             graph.remove_node(node)
        
    # for graph in list_of_graph_text:
    #     for node in graph.get_nodes():
    #         # redo the feature to make sure that it doesn't have the attributes included
    #         node.features = node.set_features_without_attributes(node.label)
    #         if node.node_type == "place":
    #             # Remove the node
    #             graph.remove_node(node)

    # # Save
    # torch.save(list_of_graph_text, 'list_of_graph_text_label_vec_no_attrib_no_room_node.pt')
    # torch.save(list_of_graph_3dssg_dict_room_label, 'list_of_graph_3dssg_dict_label_vec_no_attrib_no_room_node.pt')
    ############################################################ Removing place node and features are just the label vector, and then saving

    wandb.init(project="simplegnn",
               mode=args.mode,
            config={
                "architecture": "self attention cross attention",
                "dataset": "ScanScribe og", # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
                "epochs": args.epoch,
                "text_source": args.text_source,
                "dbscann_eps": args.dbscann_eps,
                "batch_size": args.batch_size,
                "mode": args.mode,
                "traintestsplit": args.traintestsplit,
                "seed": args.seed,
                "one_datapoint": args.one_datapoint,
                "top_k": args.top_k,
                "out_of": args.out_of,
                "sinkhorn_thr": args.sinkhorn_thr,
                "training_out_of": args.training_out_of,
                "training_top_k": args.training_top_k,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "N": args.N
            })

    # Train Test Split
    list_of_graph_text_train, list_of_graph_text_test = train_test_split(list_of_graph_text, test_size=1-args.traintestsplit)
    list_of_graph_3dssg_dict_room_label_train, list_of_graph_3dssg_dict_room_label_test = split_3dssg(list_of_graph_text_train, list_of_graph_text_test, list_of_graph_3dssg_dict_room_label)

    if (args.one_datapoint is not None):
        new_list_of_graph_text_train = []
        new_list_of_graph_3dssg_dict_room_label_train = {}
        current_scenes = set()
        for i in range(len(list_of_graph_text_train)):
            if list_of_graph_text_train[i].scene_id not in current_scenes:
                current_scenes.add(list_of_graph_text_train[i].scene_id)
                new_list_of_graph_text_train.append(list_of_graph_text_train[i])
                new_list_of_graph_3dssg_dict_room_label_train[list_of_graph_text_train[i].scene_id] = list_of_graph_3dssg_dict_room_label_train[list_of_graph_text_train[i].scene_id]

            if (len(new_list_of_graph_text_train) == args.one_datapoint):
                break

        list_of_graph_text_train = new_list_of_graph_text_train
        print("Length of list_of_graph_text_train: ", len(list_of_graph_text_train))
        list_of_graph_3dssg_dict_room_label_train = new_list_of_graph_3dssg_dict_room_label_train
    

    # Print out the data in text file format
    # delete if exists
    if os.path.exists("DATA_text.txt"):
        os.remove("DATA_text.txt")
    if os.path.exists("DATA_graph.txt"):
        os.remove("DATA_graph.txt")
    text_file = open("DATA_text.txt", "a")
    graph_file = open("DATA_graph.txt", "a")
    for t in list_of_graph_text_train:
        g = list_of_graph_3dssg_dict_room_label_train[t.scene_id]
        text_file.write(t.scene_id + "\n")
        graph_file.write(g.scene_id + "\n")
        for node in t.get_nodes():
            text_file.write(node.label + "\n")
        for node in g.get_nodes():
            graph_file.write(node.label + "\n")
        text_file.write("\n")
        graph_file.write("\n")
    text_file.close()
    graph_file.close()

    model = train_dummy_big_gnn(list_of_graph_text_train, list_of_graph_3dssg_dict_room_label_train, list_of_graph_text_test, list_of_graph_3dssg_dict_room_label_test)

    # Evaluate
    assert(len(list_of_graph_text_test) >= args.out_of)
    final_accuracy, num_datapoints = evaluate_model(model, list_of_graph_text_test, list_of_graph_3dssg_dict_room_label_test, out_of=args.out_of, top_k=args.top_k)
    # evaluate_nodes(model)
    print("Final accuracy: ", final_accuracy)
    print("Number of datapoints for calculating accuracy: ", num_datapoints)




























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