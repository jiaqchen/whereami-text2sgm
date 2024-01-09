import torch
import sys
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import DBSCAN
import copy

sys.path.insert(0, '/home/julia/Documents/h_coarse_loc/playground')
sys.path.insert(1, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
from scene_graph import SceneGraph

import spacy
import en_core_web_lg
nlp = spacy.load("en_core_web_lg")

def np_cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def nodes_features_similar(node1, node2, sim_thr=0.85):
    return np_cosine_sim(node1.features, node2.features) > sim_thr
    
def load_scene_graphs(path):
    graphs = torch.load(path)
    return graphs

def load_text_graphs(path):
    graphs = torch.load(path)
    return graphs

def combine_node_features(graph1, graph2):
    node_features1 = graph1.get_node_features()
    node_features2 = graph2.get_node_features()
    all_node_features = np.concatenate((node_features1, node_features2), axis=0)
    all_node_graph_index = np.concatenate((np.zeros(len(node_features1)), np.ones(len(node_features2))), axis=0) # graph1 is 0, graph2 is 1
    return all_node_features, all_node_graph_index

def get_matching_subgraph(graph1, graph2):
    # Cluster the nodes in both graphs with dbscan
    all_node_features, all_node_graph_index = combine_node_features(graph1, graph2)
    combined_node_idx = np.concatenate(([n1 for n1 in graph1.nodes], [n2 for n2 in graph2.nodes]), axis=0)
    assert(all([i == graph1.nodes[i].idx for i in graph1.nodes])) # key equals the idx
    assert(all([i == graph2.nodes[i].idx for i in graph2.nodes]))
    idx_mapping = {}
    for i, idx in enumerate(combined_node_idx):
        idx_mapping[i] = idx

    # Track the indices of the nodes that are matched, after combining into all_node_features
    clustering = DBSCAN(eps=0.5, min_samples=1, metric='cosine').fit(all_node_features) # default 0.05
    clusters = {}
    for i, cluster in enumerate(clustering.labels_):
        if cluster in clusters:
            clusters[cluster].append(i)
        else:
            clusters[cluster] = [i]

    # Process the clusters so that only clusters with nodes from both graphs remain
    graph1_keep_nodes = []
    graph2_keep_nodes = []
    for cluster in clusters:
        indices = clusters[cluster]
        graphs = [int(all_node_graph_index[i]) for i in indices]
        if 0 in graphs and 1 in graphs:
            graph1_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 0])
            graph2_keep_nodes.extend([idx_mapping[i] for i in indices if int(all_node_graph_index[i]) == 1])

    # Get the subgraph
    assert(type(graph1) == SceneGraph)
    assert(type(graph2) == SceneGraph)        
    graph1_keep_nodes = list(set(graph1_keep_nodes))
    graph2_keep_nodes = list(set(graph2_keep_nodes))
    subgraph1 = graph1.get_subgraph(graph1_keep_nodes, return_graph=True)
    subgraph2 = graph2.get_subgraph(graph2_keep_nodes, return_graph=True)
    # assert(subgraph1 is not None)
    # assert(subgraph2 is not None)

    # if subgraph1 is not None and subgraph2 is not None:
    #     # Print graph nodes and subgraph nodes
    #     print("Graph1 nodes: ")
    #     for nodeid in graph1.nodes:
    #         node = graph1.nodes[nodeid]
    #         print(node.label)
    #     print("Graph2 nodes: ")
    #     for nodeid in graph2.nodes:
    #         node = graph2.nodes[nodeid]
    #         print(node.label)
    #     print("Subgraph1 nodes: ")
    #     for nodeid in subgraph1.nodes:
    #         node = subgraph1.nodes[nodeid]
    #         print(node.label)
    #     print("Subgraph2 nodes: ")
    #     for nodeid in subgraph2.nodes:
    #         node = subgraph2.nodes[nodeid]
    #         print(node.label)

    return subgraph1, subgraph2

def calculate_overlap(graph1, graph2, sim_thr=0.95):
    # Go through the nodes of the two scene graphs, where overlap is calculated as out of graph1's nodes
    # how many of them are also in graph2
    if graph1 is None or graph2 is None:
        return 0
    overlap = 0
    for node1id in graph1.nodes:
        node1 = graph1.nodes[node1id]
        found_match = False
        for node2id in graph2.nodes:
            node2 = graph2.nodes[node2id]
            if nodes_features_similar(node1, node2, sim_thr):
                found_match = True
                break
        if found_match:
            overlap += 1
    return overlap / len(graph1.nodes)
