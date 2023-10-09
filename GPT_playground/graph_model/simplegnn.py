###################################### DATA ######################################

import os
import tqdm
import traceback
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, TransformerConv

import copy
from sg_dataloader import SceneGraph

from utils import print_closest_words, make_cross_graph, mask_node, accuracy_score

###################################### MODEL ######################################

class BigGNN(nn.Module):
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
        
        self.TextSelfAttention = SimpleGAT(300, 300, 300)
        self.GraphSelfAttention = SimpleGAT(300, 300, 300)
        self.TextCrossAttention = SimpleGAT(300, 300, 300)
        self.GraphCrossAttention = SimpleGAT(300, 300, 300)


        # # MLP for predicting matching score between 0 and 1
        # self.SceneText_MLP = nn.Sequential(
        #     nn.Linear(600, 600), # TODO: input dimension is hardcoded now
        #     nn.ReLU(),
        #     nn.Linear(600, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 1),
        #     nn.Sigmoid()
        # )

        # self.place_node_1_idx = place_node_1_idx # TODO: Make the place_node_idx an input to the model
        # self.place_node_2_idx = place_node_2_idx

        # Make Cross Attention Graphs
        # self.edge_index_1_cross, self.edge_attr_1_cross = make_cross_graph(x_1_dim, x_2_dim)
        # self.edge_index_2_cross, self.edge_attr_2_cross = make_cross_graph(x_2_dim, x_1_dim)
        

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

        # Batch Norm
        # x_1 = F.normalize(x_1, p=2, dim=1)
        # x_2 = F.normalize(x_2, p=2, dim=1)
        # First layer
        # x_1 = self.TextSelfAttention(x_1, edge_index_1, edge_attr_1)
        # x_2 = self.GraphSelfAttention(x_2, edge_index_2, edge_attr_2)

        for i in range(self.N):
            textSelfAttention = self.TextSelfAttentionLayers[i]
            graphSelfAttention = self.GraphSelfAttentionLayers[i]
            textCrossAttention = self.TextCrossAttentionLayers[i]
            graphCrossAttention = self.GraphCrossAttentionLayers[i]

            # Batch Norm
            # x_1 = F.normalize(x_1, p=2, dim=1)
            # x_2 = F.normalize(x_2, p=2, dim=1)

            ############# Self Attention #############
            
            x_1 = textSelfAttention(x_1, edge_index_1, edge_attr_1)
            x_2 = graphSelfAttention(x_2, edge_index_2, edge_attr_2)

            # Length of x_1 and x_2
            len_x_1 = x_1.shape[0]
            len_x_2 = x_2.shape[0]

            ############# Cross Attention #############

            # Make Cross Attention Graphs
            edge_index_1_cross, edge_attr_1_cross = make_cross_graph(x_1.shape, x_2.shape) # First half of x_1_cross should be the original x_1
            edge_index_2_cross, edge_attr_2_cross = make_cross_graph(x_2.shape, x_1.shape) # First half of x_2_cross should be the original x_2

            # Concatenate x_1 and x_2
            x_1_cross = torch.cat((x_1, x_2), dim=0)
            x_2_cross = torch.cat((x_2, x_1), dim=0)

            # Cross Attention
            x_1_cross = textCrossAttention(x_1_cross, edge_index_1_cross, edge_attr_1_cross)
            x_2_cross = graphCrossAttention(x_2_cross, edge_index_2_cross, edge_attr_2_cross)

            # Get the first len_x_1 nodes from x_1_cross
            x_1 = x_1_cross[:len_x_1] # TODO: Oh, this could be weird....... need to make sure the nodes and indices line up here
            x_2 = x_2_cross[:len_x_2]

            # norm
            x_1 = F.normalize(x_1, p=2, dim=1)
            x_2 = F.normalize(x_2, p=2, dim=1)
            

        return x_1, x_2, 0

class SimpleGAT(MessagePassing):
    # Simple one layer GATConv
    def __init__(self, in_channels_node, in_channels_edge, out_channels):
        super(SimpleGAT, self).__init__(aggr='add')  # "add" aggregation
        self.TransformerConv_nodes = TransformerConv(in_channels_node, out_channels, heads=1, concat=False, dropout=0.7)
        # self.GATConv_nodes = GATConv(in_channels_node, out_channels, heads=1, dropout=0.7)
        # self.GCNConv_nodes = GCNConv(in_channels_node, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.TransformerConv_nodes(x, edge_index)
        # x = self.GATConv_nodes(x, edge_index)

        # Take average over heads (TODO: for multi-head attention, currently not working)
        # x = x.view(x.size(0), -1, self.GATConv_nodes.heads)  # Shape: [num_nodes, out_channels, num_heads]
        # x = torch.mean(x, dim=-1)

        # layer norm
        x = F.layer_norm(x, x.shape)
        # relu
        x = F.relu(x)
        # Dropout
        # x = F.dropout(x, p=0.5, training=self.training)

        # Normalize
        x = F.normalize(x, p=2, dim=1)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    # assert(len(list_of_graph1) == len(list_of_graph2)) # TODO: not true anymore
    batch_size = 128
    for epoch in range(args.epoch):
        # for graph1, graph2 in zip(list_of_graph1, list_of_graph2):
        batch_hard_coded = 0
        for graph1 in list_of_graph1:
            graph2 = list_of_graph2_dict[graph1.scene_id]
            # Nodes
            x_1 = torch.tensor(graph1.get_node_features(), dtype=torch.float)    # Node features
            x_2 = torch.tensor(graph2.get_node_features(), dtype=torch.float)    # Node features

            if x_1.shape[0] <= 6 or x_2.shape[0] <= 6:
                continue

            # Edges
            sources_1, targets_1, features_1 = graph1.get_edge_s_t_feats()
            assert(len(sources_1) == len(targets_1) == len(features_1))
            edge_index_1 = torch.tensor([sources_1, targets_1], dtype=torch.long)
            edge_attr_1 = torch.tensor(features_1, dtype=torch.float)

            sources_2, targets_2, features_2 = graph2.get_edge_s_t_feats()
            assert(len(sources_2) == len(targets_2) == len(features_2))
            edge_index_2 = torch.tensor([sources_2, targets_2], dtype=torch.long)
            edge_attr_2 = torch.tensor(features_2, dtype=torch.float)

            # Get Place Node Index
            _, place_node_1_idx = graph1.get_place_node_idx()
            _, place_node_2_idx = graph2.get_place_node_idx()

            # Mask node
            x_1_masked, _ = mask_node(x_1, p=0.2)
            x_2_masked, _ = mask_node(x_2, p=0.2)

            # Normalize input
            x_1_masked = F.normalize(x_1_masked, p=2, dim=1)
            x_2_masked = F.normalize(x_2_masked, p=2, dim=1)

            # TRAINING STEP
            # Go through all data in one epoch
            if (batch_hard_coded % batch_size == 0):
                optimizer.zero_grad() # Clear gradients.

            out1, out2, out_matching = model(x_1_masked, x_2_masked, 
                                            edge_index_1, edge_index_2, 
                                            edge_attr_1, edge_attr_2,
                                            place_node_1_idx, place_node_2_idx) # Perform a single forward pass.
            # Cosine similarity loss
            loss1 = F.cosine_similarity(out1, x_1, dim=1) # Compute the loss.
            loss2 = F.cosine_similarity(out2, x_2, dim=1) # Compute the loss.

            # MSE loss with cosine similarity
            # loss1 = F.mse_loss(loss1, torch.tensor([1.0], dtype=torch.float)) # Compute the loss.
            # loss2 = F.mse_loss(loss2, torch.tensor([1.0], dtype=torch.float)) # Compute the loss.

            # loss3 = F.mse_loss(out_matching, torch.tensor([1.0], dtype=torch.float)) # TODO: loss3 is distance vector, but could also try cosine similarity, or after MLP
            loss3 = F.mse_loss(out1[place_node_1_idx], out2[place_node_2_idx]) # TODO: loss3 is distance vector, but could also try cosine similarity, or after MLP
            loss = ((1 - torch.cat((loss1, loss2), dim=0)).sum()) + loss3
            
            if (batch_hard_coded % batch_size == 0):
                loss.backward() # Derive gradients.
                optimizer.step() # Update parameters based on gradients.
            
            batch_hard_coded += 1

        # Print loss
        # if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # # Check accuracy
        # if epoch % 5 == 0 and epoch != 0:
        #     out1_vector = out1.detach().numpy()
        #     out2_vector = out2.detach().numpy()
        #     x_1_vector = x_1.detach().numpy()
        #     x_2_vector = x_2.detach().numpy()
        #     print_closest_words(out1_vector, x_1_vector, first_n=x_1_vector.shape[0])
        #     print()
        #     print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])

    return model

def evaluate(model):
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
            print("closest words for x_2")
            print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])
        
        i += 1

    # Print overall results
    print("Average accuracy weighted by number of nodes masked: ", sum(avg_accs) / len(avg_accs))
    print("Average accuracy weighted by number of nodes masked for x_1 (from ScanScribe): ", sum(avg_acc1s) / len(avg_acc1s))
    print("Average accuracy weighted by number of nodes masked for x_2 (from 3DSSG): ", sum(avg_acc2s) / len(avg_acc2s))


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
        x_1_masked, x_1_masked_rows = mask_node(x_1, p=0.1)
        x_2_masked, x_2_masked_rows = mask_node(x_2, p=0.1)

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
        out1, out2, out_matching = model(x_1_masked, x_2_masked, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2,
                                         place_node_1_idx, place_node_2_idx)

        out1_vector = out1.detach().numpy()
        out2_vector = out2.detach().numpy()
        x_1_vector = x_1.detach().numpy()
        x_2_vector = x_2.detach().numpy()
        # Use the masks
        out1_vector = out1_vector[x_1_masked_rows] # TODO: I think this means we are only calculating accuracy on the masked nodes, so that's good right?
        # print("out1_vector shape", out1_vector.shape)
        out2_vector = out2_vector[x_2_masked_rows]
        x_1_vector = x_1_vector[x_1_masked_rows]
        x_2_vector = x_2_vector[x_2_masked_rows]

        # print("closest words for x_1")
        # print_closest_words(out1_vector, x_1_vector, first_n=x_1_vector.shape[0])
        # print("closest words for x_2")
        # print_closest_words(out2_vector, x_2_vector, first_n=x_2_vector.shape[0])

        # Print Accuracy of just the masked nodes
        acc1 = accuracy_score(out1_vector, x_1_vector)
        acc2 = accuracy_score(out2_vector, x_2_vector)
        # print("Accuracy for x_1: ", acc1)
        # print("Accuracy for x_2: ", acc2)
        # print("Average accuracy weighted by number of nodes masked: ", (acc1 * x_1_vector.shape[0] + acc2 * x_2_vector.shape[0]) / (x_1_vector.shape[0] + x_2_vector.shape[0]))
        avg_acc = (acc1 * x_1_vector.shape[0] + acc2 * x_2_vector.shape[0]) / (x_1_vector.shape[0] + x_2_vector.shape[0])
        return acc1, acc2, avg_acc, out1_vector, out2_vector, x_1_vector, x_2_vector


if __name__ == '__main__':
    # Read -epoch parameter
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--text_source', type=str, default='human+GPT', help='human+GPT or ScanScribe3DSSG+GPT') # ScanScribe3DSSG+GPT is GPT annotated from SG, and then reparsed back into a JSON lawl
    args = parser.parse_args()
    assert(args.text_source == 'human+GPT' or args.text_source == 'ScanScribe3DSSG+GPT')

    list_of_graph_3dssg_dict = None
    list_of_graph_text = None
    list_of_graph_3dssg = None

    # We must have a list_of_graph_3dssg_dict
    if os.path.exists('list_of_graph_3dssg_dict.pt'):
        print("Using 3DSSG presaved scene graphs")
        list_of_graph_3dssg_dict = torch.load('list_of_graph_3dssg_dict.pt') 
    else: # Load 3DSSG graphs as dict
        scene_ids_3dssg = os.listdir('../../data/3DSSG/3RScan')
        list_of_graph_3dssg_dict = {}
        for scene_id in tqdm.tqdm(scene_ids_3dssg):
            try:
                scene_graph_3dssg = SceneGraph('3DSSG', scene_id, euc_dist_thres=1.0)
            except Exception as e:
                print("Error with loading 3DSSG scene graph scene ", scene_id)
                continue

            scene_graph_3dssg.to_pyg()
            scene_graph_3dssg.add_place_node() 
            list_of_graph_3dssg_dict[scene_id] = scene_graph_3dssg
        torch.save(list_of_graph_3dssg_dict, 'list_of_graph_3dssg_dict.pt')

    # Now load either ScanScribe3DSSG+GPT or human+GPT for the text source
    # 3DSSG is the set of target graphs, we use either human or GPT annotations as the text graph
    if args.text_source == 'ScanScribe3DSSG+GPT':
        print("Using ScanScribe3DSSG+GPT as text source")
        scene_ids = os.listdir('../scripts/scanscribe_json_gpt')
        
        # TODO: Try adding attributes to the features and saving another graph checkpoint
        if os.path.exists('list_of_graph_scanscribe_gpt.pt'):
            print("Using ScanScribe presaved text source")
            list_of_graph_text = torch.load('list_of_graph_scanscribe_gpt.pt')
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
            torch.save(list_of_graph_text, 'list_of_graph_scanscribe_gpt.pt')

    elif args.text_source == 'human+GPT':
        # Load Dataset (getting matched scene text pairs from human annotations)
        scene_ids = os.listdir('../output_clean')

        # Load list of graphs
        if os.path.exists('list_of_graph_human.pt') and os.path.exists('list_of_graph_3dssg.pt'):
            print("Human presaved")
            print("3DSSG presaved")
            list_of_graph_text = torch.load('list_of_graph_human.pt')
            list_of_graph_3dssg = torch.load('list_of_graph_3dssg.pt')
        else:
            # Go through folders
            list_of_graph_text = []
            list_of_graph_3dssg = []
            for scene_id in scene_ids:
                print("Loading scene ", scene_id)
                # Load scene graph
                human_folder = '../output_clean/' + scene_id
                human_subfolder = os.listdir(human_folder)[0]
                try:
                    scene_graph_human = SceneGraph('human+GPT', scene_id, raw_json='../output_clean/' + scene_id + '/' + human_subfolder + '/0_gpt_clean.json')
                except Exception as e:
                    print(e)
                    print("Error with scene ", scene_id)
                    continue
                try:
                    scene_graph_3dssg = SceneGraph('3DSSG', scene_id, euc_dist_thres=1.0)
                except Exception as e:
                    print(e)
                    print("Error with scene ", scene_id)
                    continue

                # Process graph such that there are no gaps in indices and all nodes index from 0
                scene_graph_human.to_pyg()
                scene_graph_3dssg.to_pyg()

                # Make Hierarchical node that has an edge connecting to all other nodes
                scene_graph_human.add_place_node()
                scene_graph_3dssg.add_place_node()

                # Add to list
                list_of_graph_text.append(scene_graph_human)
                list_of_graph_3dssg.append(scene_graph_3dssg)

            # Save list to file to access later
            torch.save(list_of_graph_text, 'list_of_graph_human.pt')
            torch.save(list_of_graph_3dssg, 'list_of_graph_3dssg.pt')

    if (list_of_graph_text is None or len(list_of_graph_3dssg_dict) == 0):
        print("Error loading data")
        exit()
    model = train_dummy_big_gnn(list_of_graph_text, list_of_graph_3dssg_dict)

    # Evaluate
    evaluate(model)


    # TODO: Train on a set of 10 graphs
    # TODO: Design validation metric
    #     - See if we can recover the masked nodes to a good degree
    #     - Check the vector distance between the masked nodes
    #     - Check if we recover the "word" that was masked
    # TODO: Validate on 10 examples to see if we learn anything
    # TODO: Refactor everything and make it clean to swap out with GATConv or something else
    # TODO: After refactoring, add multi-headed attention
    # TODO: Separate GCN to learn a "place" node to do comparions later

    # TODO: add relu between each transformer layer, in between self and cross attntion
    # TODO: print the accuracy alongside the loss
    # TODO: merge datapoints together
    # TODO: mask only certain nodes and make sure none overlap
    # TODO: change the loss to (binary) properly model what is supposed to be modeled?
    # TODO: train using a contrastive loss (need to set up the dataset for this)
    # TODO: clean the data to remove "floor", "wall", etc. nodes (but I think I already do this?)
    # TODO: why does Transformer work better than GATConv?
    # TODO: check that the way I calculate Loss is correct... especially with the batching
    # TODO: check the weights to see if it's actually learning something helpful??? [I'm not sure if they're changing that much?]
    # TODO: omg clean up the data my dude maybe that's why it's not working, and also I think transformers need a lot of data