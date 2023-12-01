import torch
import sys
import numpy as np
from numpy.linalg import norm

sys.path.insert(0, '/home/julia/Documents/h_coarse_loc/playground/graph_model')
import sg_dataloader
from sg_dataloader import SceneGraph
from utils import word_similarity

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

def calculate_overlap(graph1, graph2):
    # Go through the nodes of the two scene graphs, where overlap is calculated as out of graph1's nodes
    # how many of them are also in graph2
    overlap = 0
    for node1 in graph1.nodes:
        found_match = False
        for node2 in graph2.nodes:
            if nodes_features_similar(node1, node2, sim_thr=0.85):
                found_match = True
                break
        if found_match:
            overlap += 1
    return overlap / len(graph1.nodes)
