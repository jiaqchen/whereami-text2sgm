import time
import argparse
import sys
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import wandb
import random
import matplotlib.pyplot as plt

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')
from scene_graph import SceneGraph

def k_fold(dataset, folds):
    skf = KFold(folds, shuffle=True, random_state=12345)
    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset))):
        test_indices.append(torch.from_numpy(idx).to(torch.long))
    val_indices = [test_indices[i - 1] for i in range(folds)]
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def cross_entropy(preds, targets, reduction='none', dim=-1): # TODO: This could be why the loss never goes above 1
    log_softmax = torch.nn.LogSoftmax(dim=dim) 
    loss = (-targets * log_softmax(preds)).sum(1)
    assert(all(loss >= 0))
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()    

def k_fold_by_scene(dataset, folds: int):
    '''
    dataset: should be a list of SceneGraphs
    '''
    # Separate the dataset by scene
    scene_dataset = {} # mapping from scene name to list of indices from the dataset
    for i, graph in enumerate(dataset):
        if graph.scene_id not in scene_dataset:
            scene_dataset[graph.scene_id] = []
        scene_dataset[graph.scene_id].append(i)

    # Create the folds based on the scene name
    random.seed(0)
    scene_names = list(scene_dataset.keys())
    random.shuffle(scene_names)
    fold_size = len(scene_names) // folds
    train_indices, val_indices = [], []
    train_scene_names_to_check, val_scene_names_to_check = [], []
    for i in range(folds):
        val_scene_names = scene_names[i * fold_size : (i + 1) * fold_size]
        val_indices.append([idx for scene_name in val_scene_names for idx in scene_dataset[scene_name]])
        train_indices.append([idx for scene_name in scene_names if scene_name not in val_scene_names for idx in scene_dataset[scene_name]])
        val_scene_names_to_check.append(val_scene_names)
        train_scene_names_to_check.append([scene_name for scene_name in scene_names if scene_name not in val_scene_names])

    # with open(f'./training_scene_ids.txt', 'r') as f:
    #     for i, scene_name_list in enumerate(train_scene_names_to_check):
    #         assert(str(scene_name_list) == f.readline().strip())
    # with open(f'./testing_scene_ids.txt', 'r') as f:
    #     for i, scene_name_list in enumerate(val_scene_names_to_check):
    #         assert(str(scene_name_list) == f.readline().strip())

    return zip(train_indices, val_indices)
