# test for data_processing/scene_graph.py

import sys
import os
import numpy as np
import json
import torch
import random

random.seed(42)

sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
from scene_graph import SceneGraph

scene_id = '1d234004-e280-2b1a-8ec8-560046b9fc96'
test_3dssg_graph = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')[scene_id]
test_scanscribe_graph = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_edge_features.pt')[scene_id][0]

embedding_type = 'ada'

sg_3dssg = SceneGraph('3dssg', test_3dssg_graph, max_dist=1.0, embedding_type=embedding_type)
sg_scanscribe = SceneGraph('scanscribe', test_scanscribe_graph, embedding_type=embedding_type)

def test_get_subgraph_to_pyg():
    sg_3dssg_copy = SceneGraph('3dssg', test_3dssg_graph, max_dist=1.0, embedding_type=embedding_type)
    sg_scanscribe_copy = SceneGraph('scanscribe', test_scanscribe_graph, embedding_type=embedding_type)
    print(f'original edge_idx 3dssg: {sg_3dssg.edge_idx}')
    print(f'original edge_idx 3dssg: {sg_scanscribe.edge_idx}')
    # First subgraph with keeping random nodes
    keep = 0.8
    keep_nodes_3dssg = random.sample(list(sg_3dssg_copy.nodes.keys()), int(len(sg_3dssg_copy.nodes.keys())*keep))
    keep_nodes_3dssg = [int(node) for node in keep_nodes_3dssg]
    print(f'keep_nodes_3dssg: {keep_nodes_3dssg}')
    keep_nodes_scanscribe = random.sample(list(sg_scanscribe_copy.nodes.keys()), int(len(sg_scanscribe_copy.nodes.keys())*keep))
    keep_nodes_scanscribe = [int(node) for node in keep_nodes_scanscribe]
    print(f'keep_nodes_scanscribe: {keep_nodes_scanscribe}')

    subgraph_nodes_3dssg, subgraph_nodes_features_3dssg, subgraph_edges_3dssg, _ = sg_3dssg_copy.get_subgraph(keep_nodes_3dssg)
    subgraph_nodes_scanscribe, subgraph_nodes_features_scanscribe, subgraph_edges_scanscribe, _ = sg_scanscribe_copy.get_subgraph(keep_nodes_scanscribe)

    sub3dssg = SceneGraph()
    sub3dssg.nodes = subgraph_nodes_3dssg
    sub3dssg.edge_idx = subgraph_edges_3dssg
    sub3dssg.edge_features = None

    subscanscribe = SceneGraph()
    subscanscribe.nodes = subgraph_nodes_scanscribe
    subscanscribe.edge_idx = subgraph_edges_scanscribe
    subscanscribe.edge_features = None

    test_to_pyg(sub3dssg)
    test_to_pyg(subscanscribe)

def test_to_pyg(g):
    node_features, edge_ids_remap, _ = g.to_pyg() # edge_features not yet implemented
    assert(len(node_features) == len(g.nodes))
    if node_features[0] is not None: assert(len(node_features[0]) == 300 or len(node_features[0]) == 1536)
    assert(len(edge_ids_remap) == 2)
    assert(len(edge_ids_remap[0]) == len(edge_ids_remap[1]))
    assert(max(edge_ids_remap[0]) < len(node_features))
    assert(max(edge_ids_remap[1]) < len(node_features))

def test_get_subgraph():
    pass

def test_extract_edges():
    pass

def test_extract_nodes():
    pass

if __name__ == '__main__':
    test_get_subgraph_to_pyg()