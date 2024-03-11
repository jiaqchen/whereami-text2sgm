# Go through 3DSSG, ScanScribe, Human datasets, and aggregate it with the point cloud datasets from 3RScan
# and make a dataset with all the graph-pointcloud-text pairs, split it into 75% train, 25% test
# The ScanScribe and human dataset should be able to index into the 3DSSG/3RScan dataset

import os
import sys
import json
import random
import numpy as np
from tqdm import tqdm
import torch

random.seed(11)
sys.path.append('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022')

# graph dataset paths
_path_3dssg_graphs = '../data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt' # 1335 graphs
_path_scanscribe_graphs = '../data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_edge_features.pt' # 218 scenes, ? graphs
_path_human_graphs = '../data_checkpoints/processed_data/human/human_graphs_processed.pt'

# text dataset paths, for 
_path_scanscribe_text = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json' # ? texts
_path_human_text = '/home/julia/Documents/h_coarse_loc/data/human/data_json_format.json' # ? texts

# point cloud dataset paths
_path_3dssg_pc = '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/test_outputs/cells_dict.pth' # ? scenes, overlap with 3dssg?

_3dssg_graphs = torch.load(_path_3dssg_graphs)                # Dict, keys length 1335, subset of _3dssg_pc
scanscribe_graphs = torch.load(_path_scanscribe_graphs)       # Dict, keys length 218, total 4472
human_graphs = torch.load(_path_human_graphs)                 # Dict, keys length 39
scanscribe_text = json.load(open(_path_scanscribe_text, 'r')) # Dict, keys length 218, total 4472
human_text = json.load(open(_path_human_text, 'r'))           # List,      length 39
_3dssg_pc = torch.load(_path_3dssg_pc)                        # Dict, keys length 1381

# In[0]
# # randomly sample 75% of the scenes in scanscribe_graphs to be the training set
# # and the rest to be the test set
# scenes = list(scanscribe_graphs.keys())
# random.shuffle(scenes)
# train_scenes_id = scenes[:int(len(scenes)*0.75)] ############################## 75% of the scenes
# train_scenes = {scene_id: scanscribe_graphs[scene_id] for scene_id in train_scenes_id}
# test_scenes_id = scenes[int(len(scenes)*0.75):] ############################## 25% of the scenes
# test_scenes = {scene_id: scanscribe_graphs[scene_id] for scene_id in test_scenes_id}
# assert(len(train_scenes) + len(test_scenes) == len(scenes))

# # SAVE the train and test scene ids to '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/training'
# # and '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/testing'
# # write as text file
# with open('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/training/training_scene_ids.txt', 'w') as f:
#     for scene_id in train_scenes_id:
#         f.write(scene_id + '\n')
# with open('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/testing/testing_scene_ids.txt', 'w') as f:
#     for scene_id in test_scenes_id:
#         f.write(scene_id + '\n')

# # check the number of total graphs in the training and testing set
# train_scenes_total = [scene[text_id] for scene in train_scenes.values() for text_id in scene]
# print(len(train_scenes_total))

# test_scenes_total = [scene[text_id] for scene in test_scenes.values() for text_id in scene]
# print(len(test_scenes_total))

# assert(len(train_scenes_total) + len(test_scenes_total) == 4472)
# %%

# Open and read the training, testing scene_ids.txt
training_scene_ids = open('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/training/training_scene_ids.txt', 'r').read().splitlines()
testing_scene_ids = open('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/testing/testing_scene_ids.txt', 'r').read().splitlines()
assert(len(set(training_scene_ids)) == len(training_scene_ids)) # no duplicates
assert(len(set(testing_scene_ids)) == len(testing_scene_ids))


# testing_text_human_text2pos = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/training_testing_data/testing_text_human_text2pos.pt')
# testing_cells_human_text2pos = torch.load('/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/training_testing_data/testing_cells_human_text2pos.pt')
# print(len(testing_text_human_text2pos), len(testing_cells_human_text2pos))
# exit()

# In[1] Creating the different datasets FOR BASELINE TEXT2POS
def scanscribe_get_text_cells(ids, texts):
    # For the baseline Text2Pos, need a LIST of text, LIST of point clouds
    # TEXT SCANSCRIBE TRAIN
    text_scanscribe_text2pos = []
    scene_ids_for_cells = []
    for scene_id in ids:
        for text in texts[scene_id]: 
            text_scanscribe_text2pos.append(text)
            scene_ids_for_cells.append(scene_id)

    # POINT CLOUD SCANSCRIBE TRAIN
    cells_scanscribe_text2pos = []
    for scene_id in scene_ids_for_cells:
        cells_scanscribe_text2pos.append(_3dssg_pc[scene_id])
    assert(len(text_scanscribe_text2pos) == len(cells_scanscribe_text2pos))
    return text_scanscribe_text2pos, cells_scanscribe_text2pos

def human_get_text_cells(texts):
    # TEXT HUMAN
    text_human_text2pos = []
    scene_ids_for_cells_human = []
    for text in tqdm(texts): 
        text_human_text2pos.append(text['description'])
        scene_ids_for_cells_human.append(text['scanId'].split('/')[0])

    # POINT CLOUD HUMAN
    cells_human_text2pos = []
    for scene_id in tqdm(scene_ids_for_cells_human):
        cells_human_text2pos.append(_3dssg_pc[scene_id])
    assert(len(text_human_text2pos) == len(cells_human_text2pos))
    return text_human_text2pos, cells_human_text2pos

# training_text_scanscribe_text2pos, training_cells_scanscribe_text2pos = scanscribe_get_text_cells(training_scene_ids, scanscribe_text)
# testing_text_scanscribe_text2pos, testing_cells_scanscribe_text2pos = scanscribe_get_text_cells(testing_scene_ids, scanscribe_text)
testing_text_human_text2pos, testing_cells_human_text2pos = human_get_text_cells(human_text)
# save everything
prefix = '/home/julia/Documents/h_coarse_loc/baselines/Text2Pos-CVPR2022/training_testing_data'
# torch.save(training_text_scanscribe_text2pos, f'{prefix}/training_text_scanscribe_text2pos.pt')
# torch.save(training_cells_scanscribe_text2pos, f'{prefix}/training_cells_scanscribe_text2pos.pt')
# torch.save(testing_text_scanscribe_text2pos, f'{prefix}/testing_text_scanscribe_text2pos.pt')
# torch.save(testing_cells_scanscribe_text2pos, f'{prefix}/testing_cells_scanscribe_text2pos.pt')
torch.save(testing_text_human_text2pos, f'{prefix}/testing_text_human_text2pos.pt')
torch.save(testing_cells_human_text2pos, f'{prefix}/testing_cells_human_text2pos.pt')
# %%

# In[2] Creating the different datasets FOR OURS
def get_scanscribe_graphs(training_scene_ids, testing_scene_ids, scanscribe_graphs):
    scanscribe_graphs_training = {scene_id: scanscribe_graphs[scene_id] for scene_id in training_scene_ids}
    scanscribe_graphs_testing = {scene_id: scanscribe_graphs[scene_id] for scene_id in testing_scene_ids}
    return scanscribe_graphs_training, scanscribe_graphs_testing

# scanscribe_graphs_training, scanscribe_graphs_testing = get_scanscribe_graphs(training_scene_ids, testing_scene_ids, scanscribe_graphs)
# assert(len(scanscribe_graphs_training) + len(scanscribe_graphs_testing) == len(scanscribe_graphs) == 218)
# prefix = '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data'
# torch.save(scanscribe_graphs_training, f'{prefix}/training/scanscribe_graphs_train_final_no_graph_min.pt')
# torch.save(scanscribe_graphs_testing, f'{prefix}/testing/scanscribe_graphs_test_final_no_graph_min.pt')
