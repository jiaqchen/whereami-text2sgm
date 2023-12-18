# Collecting the different files for 3DSSG graphs and turning them into a single file
import os
import json
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from graph_loader_3dssg_utils import get_obj_distance, bounding_box, plot_relation

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

# Save everything as its own dictionary object
all_scenes = process_scenes('/home/julia/Documents/h_coarse_loc/data/3DSSG')

# TODO:
# [X] Validate that the bounding boxes in the scenes are correct to validate the distances are correct
