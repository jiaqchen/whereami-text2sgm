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

from playground.graph_model.data_processing.sg_dataloader import SceneGraph
from playground.graph_model.src.utils import print_closest_words, make_cross_graph, mask_node, accuracy_score, load_text_dataset

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

        # self.N_GATConv = 1 # Number of GATConv layers
        # self.gatConv = nn.ModuleList()
        # self.gatConv.append(GATConv(in_channels=D, out_channels=hidden_channels, heads=4, concat=False, dropout=0.5))
        # for _ in range(self.N_GATConv-1):
        #     self.gatConv.append(GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=4, concat=False, dropout=0.5))

        self.N_transformerConv = 1 # Number of TransformerConv layers
        self.transformerConv = nn.ModuleList()
        self.transformerConv.append(TransformerConv(in_channels=D, out_channels=hidden_channels, heads=1, concat=False, dropout=0.5)) # 16, 0.5
        for _ in range(self.N_transformerConv-1):
            self.transformerConv.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=1, concat=False, dropout=0.5))

        self.lin = Linear(hidden_channels, 4096)
        self.lin1 = Linear(4096, D)

        self.lin_text = Linear(D, hidden_channels)
        self.lin_text1 = Linear(hidden_channels, D)

    def forward(self, x_text, x_graph, edge_index, edge_attr, place_node):
        # TODO: not doing anything with x_text yet because it's already an embedding from text-embedding-ada-002
        for layer in range(self.N_transformerConv):
            x_graph = self.transformerConv[layer](x_graph, edge_index)
            x_graph = x_graph.relu()
            x_graph = F.dropout(x_graph, p=0.6, training=self.training)

        x_graph = self.lin(x_graph)
        x_graph = x_graph.relu()
        x_graph = self.lin1(x_graph)
        x_graph = x_graph.relu()
        # Take a weighted average of the nodes of graph to get global graph embedding?
        x_graph = torch.sum(x_graph, dim=0)

        x_text = self.lin_text(x_text)
        x_text = x_text.relu()
        x_text = self.lin_text1(x_text)
        x_text = x_text.relu()
        x_text = F.dropout(x_text, p=0.2, training=self.training)

        # TODO: try to output only 1 value?

        return x_text, x_graph

###################################### TRAINING ######################################

def contrastive_loss(output1, output2, label, margin=0.3):
    # label 0 for similar, 1 for dissimilar
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

def triplet_loss(output, pos, neg, m=1.0):
    # Cosine similarity, -1 dissimilar, 1 similar, between 0 and 2 for distance
    bias_term = 10
    cosine_dis_pos = 1 - F.cosine_similarity(output, pos, dim=0)
    cosine_dis_neg = 1 - F.cosine_similarity(output, neg, dim=0)
    loss = torch.mean(torch.clamp(bias_term*cosine_dis_pos - bias_term*cosine_dis_neg + m, min=0.0))
    return loss
 
def cross_entropy(preds, targets, reduction='none', dim=-1):
    log_softmax = nn.LogSoftmax(dim=dim) 
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def cosine_distance(text, graph, temperature=0.7):
    # text: (batch_size, D)
    # graph: (batch_size, D)
    # temperature: scalar
    # Returns: (batch_size, batch_size)
    text = F.normalize(text, p=2, dim=1)
    graph = F.normalize(graph, p=2, dim=1)
    logits = torch.matmul(text, graph.T)
    return 1 - logits

def train_gcn_model(text_dict: dict, graph_dict: dict):
    model = TextGCN(args.hidden_layers)#.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # TODO: no weight decay for now

    loss = torch.tensor(0.0)#.to(device)
    for epoch in range(args.epoch): # one epoch is not the entire dataset for now, so need to fix that as well
        model.train()
        scene_ids = list(text_dict.keys())
        print("scene_ids", len(scene_ids))
        # do more loops in one epoch TODO: need to change this so we go through the data smartly
        
        random.shuffle(scene_ids)
        for i in range(len(scene_ids)):
            for loop in range(args.loops_per_epoch): # TODO: change this to go through the data smartly, the looping should be on the data!
                # Get 16 scene_ids at a time
                if (i+1)*args.batch_size > len(scene_ids):
                    break
                scene_ids_batch = scene_ids[i*args.batch_size:(i+1)*args.batch_size]
                texts_batch = [torch.tensor(random.choice(text_dict[s]), dtype=torch.float) for s in scene_ids_batch]
                graphs_batch = [graph_dict[s] for s in scene_ids_batch]

                text_encoded_batch = []
                graph_encoded_batch = []
                for j in range(len(scene_ids_batch)): # batch size
                    t = texts_batch[j]
                    assert(t.shape[0] == D)
                    g = graphs_batch[j]
                    graph_features = torch.tensor(g.get_node_features(), dtype=torch.float)#.to(device)
                    sources, targets, edge_features = g.get_edge_s_t_feats()
                    edge_indices = torch.tensor([sources, targets], dtype=torch.long)
                    edge_attr = torch.tensor(edge_features, dtype=torch.float)
                    _, place_node = g.get_place_node_idx()
                    text_encoded, graph_encoded = model(t, graph_features, edge_indices, edge_attr, place_node)
                    text_encoded_batch.append(text_encoded)
                    graph_encoded_batch.append(graph_encoded)

                optimizer.zero_grad()

                # Calculate logits
                text_encoded_batch = torch.stack(text_encoded_batch, dim=0)
                graph_encoded_batch = torch.stack(graph_encoded_batch, dim=0)
                assert(text_encoded_batch.shape[0] == args.batch_size)
                assert(graph_encoded_batch.shape[0] == args.batch_size)
                assert(text_encoded_batch.shape[1] == D)
                assert(graph_encoded_batch.shape[1] == D)
                # Make the logits the cosine distance between the text and graph embeddings as a matrix
                # Shape: (batch_size, batch_size)
                logits = cosine_distance(text_encoded_batch, graph_encoded_batch) 
                print("logits", logits)
                assert(logits.shape[0] == args.batch_size)
                assert(logits.shape[1] == args.batch_size)

                target = torch.zeros(args.batch_size, args.batch_size)#.to(device)
                target = target + 2*torch.eye(args.batch_size)#.to(device)

                # Cross entropy
                loss1 = cross_entropy(logits, target, reduction='mean', dim=1)
                loss2 = cross_entropy(logits.T, target.T, reduction='mean', dim=1)
                loss = (loss1 + loss2) / 2.0
                loss = loss.mean()


                # texts_similarity = text_encoded_batch @ text_encoded_batch.T # Shape: (batch_size, batch_size)
                # graph_similarity = graph_encoded_batch @ graph_encoded_batch.T
                # targets = F.softmax((texts_similarity + graph_similarity) / 2 * args.temperature, dim=-1)

                # texts_loss = cross_entropy(logits, targets, reduction='none')
                # graphs_loss = cross_entropy(logits.T, targets.T, reduction='none')
                # loss = (texts_loss + graphs_loss) / 2.0 # Shape: (batch_size, batch_size)
                # loss = loss.mean()
                wandb.log({"loss_per_batch": loss.item()})
                # Bias term?
                loss.backward()
                optimizer.step()

                accuracy_val = evaluate_classification(model, dict_3dssg_ada_labels_val, text=dict_of_texts_val, top_k=args.top_k) # Accuracy on validation set
                wandb.log({"accuracy_val_per_batch": accuracy_val})
                model.train() # Reset model to train mode

    return model

###################################### EVALUATION ######################################

def evaluate_classification(model, _3dssg, text=None, num_eval=10, top_k=10, sample_size=50):
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
        
        graph_embeddings = []
        text_embeddings = []
        for scene_id in list(text.keys()):
            g = _3dssg[scene_id]
            graph_features = torch.tensor(g.get_node_features(), dtype=torch.float)#.to(device)
            sources, targets, edge_features = g.get_edge_s_t_feats()
            edge_indices = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            _, place_node = g.get_place_node_idx()
            text_embedding, graph_embedding = model(torch.tensor(random.choice(text[scene_id]), dtype=torch.float), 
                                                    graph_features, edge_indices, edge_attr, place_node) # TODO: dummy uses random text embedding
            graph_embeddings.append(graph_embedding)
            text_embeddings.append(text_embedding)

        # Calculate accuracy
        accuracy = []
        for i in range(len(text_embeddings)):
            text_embedding = text_embeddings[i]
            similarities = []
            for j in range(len(graph_embeddings)):
                graph_embedding = graph_embeddings[j]

                ## Cross entropy score
                # text_similarity = F.softmax(text_embedding @ graph_embedding.T, dim=-1)
                # graph_similarity = F.softmax(graph_embedding @ text_embedding.T, dim=-1)
                # similarity = (text_similarity + graph_similarity) / 2.0
                # similarities.append(similarity)
                # Normalize both
                text_similarity = F.normalize(text_embedding, p=2, dim=0)
                graph_similarity = F.normalize(graph_embedding, p=2, dim=0)
                # dot
                similarity = torch.dot(text_similarity, graph_similarity)
                similarities.append(similarity)

            # Get the top_k most similar scene graphs
            similarities = torch.tensor(similarities, dtype=torch.float)
            top_k_similar = torch.topk(similarities, k=top_k, dim=-1)
            top_k_similar = top_k_similar.indices.tolist() # TODO: check that the indices are correct
            if i in top_k_similar:
                accuracy.append(1)
            else:
                accuracy.append(0)

        return sum(accuracy) / len(accuracy)



###################################### LOAD DATASETS ######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--train_test_split', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--triplossmargin', type=float, default=None)
    parser.add_argument('--hidden_layers', type=int, default=512)
    parser.add_argument('--force_retrain', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--loops_per_epoch', type=int, default=1)
    args = parser.parse_args()

    random.seed(args.seed)

    dict_of_texts = None
    dict_of_texts_train = None
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

    if (args.train_test_split is not None):
        keys = list(dict_of_texts.keys())
        random.shuffle(keys)
        train_size = args.train_test_split * len(keys) # Separate based off train/test percentage
        keys_train = keys[0:int(train_size)]
        validation_keys = keys[int(train_size):] # Use the rest for validation
        dict_of_texts_train = {k: dict_of_texts[k] for k in keys_train}
        dict_3dssg_ada_labels_train = {k: dict_3dssg_ada_labels[k] for k in keys_train}
        dict_of_texts_val = {k: dict_of_texts[k] for k in validation_keys}
        dict_3dssg_ada_labels_val = {k: dict_3dssg_ada_labels[k] for k in validation_keys}
        assert(len(dict_of_texts_train) == len(dict_3dssg_ada_labels_train))
        assert(len(dict_of_texts_val) == len(dict_3dssg_ada_labels_val))
        assert(len(dict_of_texts_val) > 0)

    using_triploss = args.triplossmargin != None
    wandb.init(mode=args.mode,
            project="text-gcn",
            config={
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "architecture": "TransformerConv+Linear",
                "dataset": "ScanScribe_1", # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
                "epochs": args.epoch,
                "train_test_split": args.train_test_split,
                "triplet_loss": using_triploss,
                "trip_cross_ent": False,
                "triplet_loss_margin": args.triplossmargin,
                "hidden_layers": args.hidden_layers,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "temperature": args.temperature,
                "notes": "taking avg over graph nodes, 2 linear layers for text, cross entropy version with logits"
            })

    if args.force_retrain:
        model = train_gcn_model(dict_of_texts_train, dict_3dssg_ada_labels_train)
        torch.save(model.state_dict(), 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_traintestsplit_'+str(args.train_test_split)+'_hiddenlayers_'+str(args.hidden_layers)+'_batchsize_'+str(args.batch_size)+'.pt')
    elif os.path.exists('text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_traintestsplit_'+str(args.train_test_split)+'_hiddenlayers_'+str(args.hidden_layers)+'_batchsize_'+str(args.batch_size)+'.pt'):
        print('Loading saved model ' + 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_traintestsplit_'+str(args.train_test_split)+'_hiddenlayers_'+str(args.hidden_layers)+'_batchsize_'+str(args.batch_size)+'.pt')
        model = TextGCN()
        model.load_state_dict(torch.load('text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_traintestsplit_'+str(args.train_test_split)+'_hiddenlayers_'+str(args.hidden_layers)+'_batchsize_'+str(args.batch_size)+'.pt'))
        model.eval()
    else:
        model = train_gcn_model(dict_of_texts_train, dict_3dssg_ada_labels_train)
        torch.save(model.state_dict(), 'text_gcn_epoch_'+str(args.epoch)+'_tripletloss_'+str(args.triplossmargin)+'_traintestsplit_'+str(args.train_test_split)+'_hiddenlayers_'+str(args.hidden_layers)+'_batchsize_'+str(args.batch_size)+'.pt')

    # Evaluate
    pos, neg, total = evaluate_classification(model, dict_3dssg_ada_labels, text=dict_of_texts_val)
    print("Accuracy on positive pairs: ", pos)
    print("Accuracy on negative pairs: ", neg)
    print("Accuracy on all pairs: ", total)