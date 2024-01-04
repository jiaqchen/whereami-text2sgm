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
            scene_dir = sorted([int(x[:-5]) for x in os.listdir(os.path.join(dir_to_scenes, id))])
            scene = {}
            for file in scene_dir:
                filename = str(file) + '.json'
                with open(os.path.join(dir_to_scenes, id, filename)) as f:
                    data = f.read()
                    data = txt_to_json(data)
                    data = json.loads(data)
                    scene[file] = data
        except Exception as e:
            print(f'Error processing scene {id}: {e}')

        scenes[id] = scene
    return scenes
                
def add_edge_features(all_scenes):
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for txt_id in all_scenes[scene_id]:
            for edge in all_scenes[scene_id][txt_id]['edges']:
                edge['relation_ada'], hada = get_ada(edge['relationship'], hada)
                edge['relation_word2vec'], hw2v = get_word2vec(edge['relationship'], hw2v)
                # NOTE: assumed there are no attributes for edges
    return all_scenes

def add_node_features(all_scenes):
    hada = {}
    hw2v = {}
    for scene_id in tqdm(all_scenes):
        for txt_id in all_scenes[scene_id]:
            for node in all_scenes[scene_id][txt_id]['nodes']:
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
    # all_scenes = process_scenes_to_dict('/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned')
    # all_scenes = torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/raw_data/scanscribe/scanscribe_cleaned_original.pt')

    # all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/raw_data/scanscribe/scanscribe_cleaned_original.pt')
    # all_scenes = add_node_features(all_scenes)
    # torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_features.pt')

    # all_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_features.pt')
    # all_scenes = add_edge_features(all_scenes)
    # torch.save(all_scenes, '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_edge_features.pt')
    
    p = '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_edge_features.pt'
    all_scenes = torch.load(p)

    print(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['edges'][1]['relationship'])
    print(len(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['edges'][1]['relation_ada']))
    print(len(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['edges'][1]['relation_word2vec']))
    print(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['nodes'][1]['attributes'])
    print(len(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['nodes'][1]['attributes_ada']['all']))
    print(len(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['nodes'][1]['attributes_word2vec']['all']))

    print(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['nodes'][1].keys())
    print(all_scenes['1d234004-e280-2b1a-8ec8-560046b9fc96'][0]['edges'][1].keys())