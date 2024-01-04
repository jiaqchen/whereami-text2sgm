# Collecting the different files for 3DSSG graphs and turning them into a single file
import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_loader_utils import get_obj_distance, bounding_box, plot_relation, get_ada, get_word2vec

# Parameters for creating scene graphs
dist_thr = 1.0

# Load the files once
objects_path = '/home/julia/Documents/h_coarse_loc/data/3DSSG/objects.json'
objects = {}
with open(objects_path, 'r') as f:
    objects = json.load(f)
scans = objects['scans']
scans_dict = {}
for s in scans:
    scans_dict[s['scan']] = s

relationships_path = '/home/julia/Documents/h_coarse_loc/data/3DSSG/relationships.json'
relationships = {}
with open(relationships_path, 'r') as f:
    relationships = json.load(f)
relationships = relationships['scans']
relationships_dict = {}
for r in relationships:
    relationships_dict[r['scan']] = r

def process_scenes(dir_to_scenes, plot=False, dist_thr=1.0):
    ids = os.listdir(os.path.join(dir_to_scenes, '3RScan'))
    scenes = {}
    for id in tqdm(ids):
        print(f'Processing scene {id}')
        try:
            scene = {}
            scene['objects'], scene['relationships'] = process_objects_and_relationships(dir_to_scenes, id, plot, dist_thr)
            assert(id not in scenes)
            scenes[id] = scene
        except Exception as e:
            print(f'Error processing scene {id}: {e}')
        
    assert(len(scenes) == len(relationships_dict))
    return scenes

def process_objects_and_relationships(dir_to_objects, scene_id, plot=False, dist_thr=1.0):
    # Path for segmentations
    segmentations_path = os.path.join(dir_to_objects, '3RScan', scene_id, 'semseg.v2.json')
    segmentations = {}
    with open(segmentations_path, 'r') as f:    
        segmentations = json.load(f)
    segmentations = segmentations['segGroups']

    objects_in_scan = scans_dict[scene_id]['objects']
    assert len(objects_in_scan) == len(segmentations)

    objects_in_scan = pd.DataFrame(objects_in_scan)
    objects_in_scan = objects_in_scan.drop(columns=['nyu40', 'ply_color', 'eigen13', 'rio27', 'affordances', 'state_affordances', 'symmetry'], errors='ignore') # Ignore errors if column does not exist

    # Convert back to list of dicts
    objects_in_scan = objects_in_scan.set_index('id', drop=False).to_dict('index')

    # Turn segmentation into a dict indexted by id
    segmentations_dict = {}
    for seg in segmentations:
        segmentations_dict[str(seg['id'])] = seg

    graph_adj = {}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Add segmentation to objects_in_scan per object
    for obj_id in objects_in_scan.keys():
        obj = objects_in_scan[obj_id]
        # Check that we are adding the correct segmentation to the correct object based on label
        assert obj['label'] == segmentations_dict[obj_id]['label']
        obj['obb'] = segmentations_dict[obj['id']]['obb']

        if (plot):
            bounding_box(obj, ax, plot=True)

        # Also add object to graph_adj
        graph_adj[obj_id] = {'label': obj['label'], 'adj_to': []}

    # Process relationships
    assert(scene_id in relationships_dict)
    relationship = relationships_dict[scene_id]

    for r in relationship['relationships']:
        # first_obj = str(r[0])  # str(int)
        # second_obj = str(r[1]) # str(int)
        # relation = r[3]   # str; first_obj --> relation --> second_obj
        #                     # side table 'standing on' floor
        assert(str(r[0]) in graph_adj)
        assert(str(r[1]) in graph_adj)

        # Add to adjacency list
        distance = get_obj_distance(str(r[0]), str(r[1]), objects_in_scan)
        if plot and distance < dist_thr:
            plot_relation(objects_in_scan[str(r[0])], objects_in_scan[str(r[1])], ax, distance)

        graph_adj[str(r[0])]['adj_to'].append({'obj_id': str(r[1]), 'relation': r[3], 'distance': distance})

    if plot:
        plt.show()
        
    return objects_in_scan, graph_adj

def add_edge_list(all_scenes):
    hada = {}
    hw2v = {}
    for sceneid in tqdm(all_scenes):
        relationships = relationships_dict[sceneid]['relationships']
        obj1_list = []
        obj2_list = []
        relation_list = []
        relation_word2vec_list = []
        relation_ada_list = []
        dist_list = []
        for rel in relationships:
            obj1_list.append(rel[0])
            obj2_list.append(rel[1])
            relation_list.append(rel[3])
            w2v, hw2v = get_word2vec(rel[3], hw2v)
            relation_word2vec_list.append(w2v)
            ada, hada = get_ada(rel[3], hada)
            relation_ada_list.append(ada)
            dist_list.append(get_obj_distance(str(rel[0]), str(rel[1]), all_scenes[sceneid]['objects']))
        assert(len(obj1_list) == len(obj2_list) == len(relation_list) == len(dist_list))
        all_scenes[sceneid]['edge_lists'] = {}
        all_scenes[sceneid]['edge_lists']['from'] = obj1_list
        all_scenes[sceneid]['edge_lists']['to'] = obj2_list
        all_scenes[sceneid]['edge_lists']['relation'] = relation_list
        all_scenes[sceneid]['edge_lists']['relation_word2vec'] = relation_word2vec_list
        all_scenes[sceneid]['edge_lists']['relation_ada'] = relation_ada_list
        all_scenes[sceneid]['edge_lists']['distance'] = dist_list
    torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt') # uncomment to save

def add_node_features(all_scenes):
    hada = {}
    hw2v = {}
    print(len(all_scenes))
    for scene in tqdm(all_scenes):
        objects = all_scenes[scene]['objects']
        for obj in tqdm(objects):
            label_ada, hada = get_ada(objects[obj]['label'], hada)
            objects[obj]['label_ada'] = label_ada
            label_word2vec, hw2v = get_word2vec(objects[obj]['label'], hw2v)
            objects[obj]['label_word2vec'] = label_word2vec
            attributes_word2vec = {}
            attributes_ada = {}
            for attrs in objects[obj]['attributes']:
                attributes_word2vec[attrs] = []
                attributes_ada[attrs] = []
                for attr in objects[obj]['attributes'][attrs]:
                    attr_word2vec, hw2v = get_word2vec(attr, hw2v)
                    attributes_word2vec[attrs].append(attr_word2vec)
                    attr_ada, hada = get_ada(attr, hada)
                    attributes_ada[attrs].append(attr_ada)
            objects[obj]['attributes_word2vec'] = attributes_word2vec
            objects[obj]['attributes_ada'] = attributes_ada
    torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed.pt') # uncomment to save

def check_num_edges(all_scenes):
    num_edges = []
    adj_num_edges = []
    for scene in all_scenes:
        num_edges.append(len(all_scenes[scene]['edge_lists']['from']))
        adj_list = all_scenes[scene]['relationships']
        scene_sum = 0
        for adj in adj_list:
            adj_to = len(adj_list[adj]['adj_to'])
            scene_sum += adj_to
        adj_num_edges.append(scene_sum)

    assert(len(num_edges) == len(adj_num_edges))
    assert(all(num_edges[i] == adj_num_edges[i] for i in range(len(num_edges))))

def change_w2v_word2vec(all_scenes, p):
    for scene_id in tqdm(all_scenes):
        for node_id in all_scenes[scene_id]['objects']:
            node = all_scenes[scene_id]['objects'][node_id]
            node['attributes_word2vec'] = node['attributes_w2v']
            del node['attributes_w2v']
    torch.save(all_scenes, p)


if __name__ == "__main__":
    all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/raw_data/3dssg/3dssg_graphs_original.pt')
    add_node_features(all_scenes)
    all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed.pt')
    add_edge_list(all_scenes)

    # p = '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt'
    # all_scenes = torch.load(p)
    # change_w2v_word2vec(all_scenes, p)

    # p = '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed.pt'
    # all_scenes = torch.load(p)
    # change_w2v_word2vec(all_scenes, p)
