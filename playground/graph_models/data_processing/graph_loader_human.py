import os
import json
import torch
from tqdm import tqdm
import re

from graph_loader_utils import get_ada, get_word2vec, check_and_remove_invalid_edges
from graph_models.src.utils import txt_to_json

def process_scenes_to_dict(dir_to_scenes):
    ids = os.listdir(dir_to_scenes)
    scenes = {}
    for id in tqdm(ids):
        # print(f'Processing scene {id}')
        try:
            # scene_dir = sorted([int(x[:-5]) for x in os.listdir(os.path.join(dir_to_scenes, id))])
            scene = {}
            filename = id
            with open(os.path.join(dir_to_scenes, filename)) as f:
                scene = json.load(f)
        except Exception as e:
            print(f'Error processing scene {id}: {e}')

        scenes[id.split('.')[0]] = scene
    return scenes
                
def add_edge_features(all_scenes):
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for edge in all_scenes[scene_id]['edges']:
            edge['relation_ada'], hada = get_ada(edge['relationship'], hada)
            edge['relation_word2vec'], hw2v = get_word2vec(edge['relationship'], hw2v)
            # NOTE: assumed there are no attributes for edges
    return all_scenes

def add_node_features(all_scenes):
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for node in all_scenes[scene_id]['nodes']:
            node['label_ada'], hada = get_ada(node['label'], hada)
            node['label_word2vec'], hw2v = get_word2vec(node['label'], hw2v)
            attributes_ada = {'all': []}
            attributes_word2vec = {'all': []}
            for attribute in node['attributes']:
                attr_ada, hada = get_ada(attribute, hada)
                attr_word2vec, hw2v = get_word2vec(attribute, hw2v)
                attributes_ada['all'].append(attr_ada)
                attributes_word2vec['all'].append(attr_word2vec)
            node['attributes_ada'] = attributes_ada
            node['attributes_word2vec'] = attributes_word2vec
    return all_scenes

if __name__ == '__main__':
    all_scenes = process_scenes_to_dict('/home/julia/Documents/h_coarse_loc/data/human/data_extract_completion')
    # all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/human/human_graphs_unprocessed.pt')
    all_scenes = add_node_features(all_scenes)
    all_scenes = add_edge_features(all_scenes)
    torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/human/human_graphs_processed.pt')

    all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/human/human_graphs_processed.pt')
    print(f'keys: {all_scenes.keys()}')
    print(f'len of keys: {len(all_scenes.keys())}')