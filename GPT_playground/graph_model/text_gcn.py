###################################### DATA ######################################

import copy
import wandb
import random
import json
import argparse
import os
import tqdm
import traceback
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GCNConv, TransformerConv
import clip

from sg_dataloader import SceneGraph
from utils import print_closest_words, make_cross_graph, mask_node, accuracy_score, load_text_dataset

###################################### MODEL ######################################

D = 1536 # text and graph embedding dimension
device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model, preprocess = clip.load("ViT-B/32", device=device) # TODO: may need to try different models
dict_of_texts_val = None
dict_3dssg_ada_labels_val = None

class TextGCN(nn.Module): # Not really a GCN, but a GAT?? Using TransformerConv

    def __init__(self, hidden_channels):
        super().__init__()
        # Frozen text transformer (CLIP Text Transformer)
        # To output a D-dim text embedding
        # NOTE: I think I can directly put this in the forward pass as a clip_model.encode_text() call

        # GCN to learn graph embedding, also D-dim, global graph embedding
        # NOTE: We're just going to try a SimpleGAT for now
        self.transformerConv1 = TransformerConv(in_channels=D, out_channels=hidden_channels, 
                                               heads=8, concat=False, 
                                               dropout=0.7) # TODO: one layer for now
        self.transformerConv2 = TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels,
                                                  heads=8, concat=False,
                                                    dropout=0.7)
        self.lin = Linear(hidden_channels, D)

    def forward(self, x_text, x_graph, edge_index, edge_attr, place_node):
        # TODO: not doing anything with x_text yet because it's already an embedding from text-embedding-ada-002
        x_graph = self.transformerConv1(x_graph, edge_index)
        x_graph = x_graph.relu()
        x_graph = self.lin(x_graph)
        x_graph = torch.mean(x_graph, dim=0)

        return x_text, x_graph

###################################### TRAINING ######################################

def contrastive_loss(output1, output2, label, margin=0.3):
    # label 0 for similar, 1 for dissimilar
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

def triplet_loss(output, pos, neg, m=1.0):
    # pos_dist = F.pairwise_distance(output, pos, keepdim=True)
    # neg_dist = F.pairwise_distance(output, neg, keepdim=True)
    # # print("pos_dist: ", pos_dist, " neg_dist: ", neg_dist)
    # loss = torch.mean(torch.clamp(pos_dist - neg_dist + m, min=0.0))
    # return loss
    # Cosine similarity, -1 dissimilar, 1 similar, between 0 and 2 for distance
    cosine_dis_pos = 1 - F.cosine_similarity(output, pos, dim=0)
    cosine_dis_neg = 1 - F.cosine_similarity(output, neg, dim=0)
    # print("cosine_dis_pos: ", cosine_dis_pos, " cosine_dis_neg: ", cosine_dis_neg)
    loss = torch.mean(torch.clamp(cosine_dis_pos - cosine_dis_neg + m, min=0.0))
    return loss

def train_gcn_model(text_dict: dict, graph_dict: dict):
    model = TextGCN(args.hidden_layers)#.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # TODO: no weight decay for now

    loss = torch.tensor(0.0)#.to(device)
    batch = 128
    for epoch in range(args.epoch):
        model.train()
        scene_ids = text_dict.keys()
        for s in scene_ids:
            texts = text_dict[s]
            for t in texts:
                graph = graph_dict[s]
                # Get negative graph
                graph_keys = list(graph_dict.keys())
                graph_keys.remove(s)
                graph_neg = graph_dict[random.choice(graph_keys)]

                graph_features = torch.tensor(graph.get_node_features(), dtype=torch.float)#.to(device)
                sources, targets, edge_features = graph.get_edge_s_t_feats()
                edge_indices = torch.tensor([sources, targets], dtype=torch.long)#.to(device)
                edge_attr = torch.tensor(edge_features, dtype=torch.float)#.to(device)
                _, place_node = graph.get_place_node_idx()

                graph_features_neg = torch.tensor(graph_neg.get_node_features(), dtype=torch.float)#.to(device)
                sources_neg, targets_neg, edge_features_neg = graph_neg.get_edge_s_t_feats()
                edge_indices_neg = torch.tensor([sources_neg, targets_neg], dtype=torch.long)#.to(device)
                edge_attr_neg = torch.tensor(edge_features_neg, dtype=torch.float)#.to(device)
                _, place_node_neg = graph_neg.get_place_node_idx()

                if batch == 128:
                    optimizer.zero_grad()

                text_encoded, graph_encoded = model(t, graph_features, edge_indices, edge_attr, place_node)
                _, graph_encoded_neg = model(t, graph_features_neg, edge_indices_neg, edge_attr_neg, place_node_neg)
                text_encoded = torch.tensor(text_encoded, dtype=torch.float)#.to(device) # TODO: remove in future if not needed
                # loss1 = F.pairwise_distance(graph_encoded, text_encoded)
                # loss2 = F.pairwise_distance(graph_encoded_neg, text_encoded)
                loss1 = 1 - F.cosine_similarity(graph_encoded, text_encoded, dim=0)
                loss += triplet_loss(text_encoded, graph_encoded, graph_encoded_neg, m=args.triplossmargin) + loss1
                # loss = F.mse_loss(graph_encoded, text_encoded)
                if batch == 128:
                    loss = loss / batch
                    loss.backward()
                    optimizer.step()
                    wandb.log({"loss_per_batch": loss.item()})
                    batch = 0
                    loss = torch.tensor(0.0)#.to(device)
                else:
                    batch += 1
            
            
        loss_rounded = round(loss.item(), 10)
        accuracy = evaluate_classification(model, graph_dict, text=text_dict) # Accuracy on train set
        accuracy_val = evaluate_classification(model, dict_3dssg_ada_labels_val, text=dict_of_texts_val) # Accuracy on validation set
        # Only print decimals up to 6 places
        accuracy_rounded = [round(a, 6) for a in accuracy]
        accuracy_val_rounded = [round(a, 6) for a in accuracy_val]
        wandb.log({"loss_per_epoch": loss_rounded, 
                        "accuracy_train_pos": accuracy_rounded[0],
                        "accuracy_train_neg": accuracy_rounded[1],
                        "accuracy_train": accuracy_rounded[2],
                        "accuracy_val_pos": accuracy_val_rounded[0],
                        "accuracy_val_neg": accuracy_val_rounded[1],
                        "accuracy_val": accuracy_val_rounded[2]})

        print("Epoch: " + str(epoch) + ", Loss: " + str(loss_rounded) + ", Accuracy: " + str(accuracy_rounded) + ", Accuracy Val: " + str(accuracy_val_rounded))

        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt')


    return model

###################################### EVALUATION ######################################

def evaluate_classification(model, _3dssg, text=None, num_eval=10):
    model.eval()
    with torch.no_grad():
        # Use human+GPT text and 3DSSG graph
        if text is None:
            # Use human+GPT
            # Get the human+GPT texts
            # Find corresponding _3dssg graph
            # For each pair, pass through model and get a classification
            # Calculate accuracy
            return

        accuracies = []
        accuracies_neg = []
        keys = list(dict_of_texts.keys())
        for scene_id in keys[0:num_eval]: # Only do num_eval scenes
            ts = dict_of_texts[scene_id]
            for t in ts:
                graph = _3dssg[scene_id]
                # Get negative graph
                graph_keys = list(_3dssg.keys())
                graph_keys.remove(scene_id)
                graph_neg = _3dssg[random.choice(graph_keys)]

                graph_features = torch.tensor(graph.get_node_features(), dtype=torch.float)#.to(device)
                sources, targets, edge_features = graph.get_edge_s_t_feats()
                edge_indices = torch.tensor([sources, targets], dtype=torch.long)
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
                _, place_node = graph.get_place_node_idx()

                graph_features_neg = torch.tensor(graph_neg.get_node_features(), dtype=torch.float)#.to(device)
                sources_neg, targets_neg, edge_features_neg = graph_neg.get_edge_s_t_feats()
                edge_indices_neg = torch.tensor([sources_neg, targets_neg], dtype=torch.long)
                edge_attr_neg = torch.tensor(edge_features_neg, dtype=torch.float)
                _, place_node_neg = graph_neg.get_place_node_idx()

                text_encoded, graph_encoded = model(t, graph_features, edge_indices, edge_attr, place_node)
                text_encoded = torch.tensor(text_encoded, dtype=torch.float)
                # dist = F.pairwise_distance(graph_encoded, text_encoded)
                dist = 1 - F.cosine_similarity(graph_encoded, text_encoded, dim=0)

                text_encoded_neg, graph_encoded_neg = model(t, graph_features_neg, edge_indices_neg, edge_attr_neg, place_node_neg)
                text_encoded_neg = torch.tensor(text_encoded_neg, dtype=torch.float)
                # dist_neg = F.pairwise_distance(graph_encoded_neg, text_encoded_neg)
                dist_neg = 1 - F.cosine_similarity(graph_encoded_neg, text_encoded_neg, dim=0)
                # print("dist: ", dist.item(), " dist_neg: ", dist_neg.item())

                if (dist.item() < 1): accuracies.append(1) # < 1 because cosine similarity middle distance is 1
                else: accuracies.append(0)
                if (dist_neg.item() < 1): accuracies_neg.append(0)
                else: accuracies_neg.append(1)

    return sum(accuracies)/len(accuracies), sum(accuracies_neg)/len(accuracies_neg), sum(accuracies+accuracies_neg)/(len(accuracies+accuracies_neg))

###################################### LOAD DATASETS ######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--smaller_dataset', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--triplossmargin', type=float, default=None)
    parser.add_argument('--hidden_layers', type=int, default=512)
    parser.add_argument('--force_retrain', type=bool, default=False)
    args = parser.parse_args()

    dict_of_texts = None
    dict_3dssg_ada_labels = None

    # Input to the model is the text directly and the graph
    # text_scan_ids, dict_of_texts = load_text_dataset()
    if os.path.exists('human+GPT_cleaned_text_embedding_ada_002.pt'):
        dict_of_texts = torch.load('human+GPT_cleaned_text_embedding_ada_002.pt')

    # Load the scene graph dataset
    if os.path.exists('dict_3dssg_ada_labels.pt'):
        print("Using 3DSSG presaved scene graphs")
        dict_3dssg_ada_labels = torch.load('dict_3dssg_ada_labels.pt') 
    else: # Load 3DSSG graphs as dict
        scene_ids_3dssg = os.listdir('../../data/3DSSG/3RScan')
        dict_3dssg_ada_labels = {}
        for scene_id in tqdm.tqdm(scene_ids_3dssg):
            try:
                scene_graph_3dssg = SceneGraph('3DSSG', scene_id, euc_dist_thres=1.0)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                print("Error with loading 3DSSG scene graph scene ", scene_id)
                continue
            try:
                scene_graph_3dssg.to_pyg()
            except:
                continue
            scene_graph_3dssg.add_place_node() 
            dict_3dssg_ada_labels[scene_id] = scene_graph_3dssg
        torch.save(dict_3dssg_ada_labels, 'dict_3dssg_ada_labels.pt')

    if (dict_of_texts is None or len(dict_3dssg_ada_labels) == 0):
        print("Error with loading datasets")
        exit()

    if (args.smaller_dataset is not None):
        # Select a subset of the dataset
        keys = list(dict_of_texts.keys())
        random.shuffle(keys)
        keys = keys[0:args.smaller_dataset]
        validation_keys = keys[-args.smaller_dataset:]
        dict_of_texts = {k: dict_of_texts[k] for k in keys}
        dict_3dssg_ada_labels = {k: dict_3dssg_ada_labels[k] for k in keys}
        dict_of_texts_val = {k: dict_of_texts[k] for k in validation_keys}
        dict_3dssg_ada_labels_val = {k: dict_3dssg_ada_labels[k] for k in validation_keys}
        assert(len(dict_of_texts) == len(dict_3dssg_ada_labels))
        assert(len(dict_of_texts_val) == len(dict_3dssg_ada_labels_val))
        assert(len(dict_of_texts) == args.smaller_dataset)

    using_triploss = args.triplossmargin != None
    wandb.init(project="text-gcn",
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "architecture": "TransformerConv+Linear",
                "dataset": "ScanScribe_1",
                "epochs": args.epoch,
                "smaller_dataset": args.smaller_dataset,
                "triplet_loss": using_triploss,
                "trip_cross_ent": False,
                "triplet_loss_margin": args.triplossmargin,
                "hidden_layers": args.hidden_layers,
                "notes": "taking avg over graph nodes, pairwise distance for accuracy"
            })

    if args.force_retrain:
        model = train_gcn_model(dict_of_texts, dict_3dssg_ada_labels)
        torch.save(model.state_dict(), 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt')
    elif os.path.exists('text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt'):
        print('Loading saved model ' + 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt')
        model = TextGCN()
        model.load_state_dict(torch.load('text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt'))
        model.eval()
    else:
        model = train_gcn_model(dict_of_texts, dict_3dssg_ada_labels)
        torch.save(model.state_dict(), 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_smallerdataset_'+str(args.smaller_dataset)+'_hiddenlayers_'+str(args.hidden_layers)+'.pt')

    # Evaluate
    pos, neg, total = evaluate_classification(model, dict_3dssg_ada_labels, text=dict_of_texts)
    print("Accuracy on positive pairs: ", pos)
    print("Accuracy on negative pairs: ", neg)
    print("Accuracy on all pairs: ", total)