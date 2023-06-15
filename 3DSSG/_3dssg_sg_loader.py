# Class for scene graph loader

import os
import json
import numpy as np

class SceneGraph:
    def __init__(self, scene_id, euc_dist_thres=1.0):
        self.scene_id = scene_id
        self.euc_dist_thres = euc_dist_thres # For adding edges to the graph

        self.scene_graph = self.load_scene_graph()
        self.objects_in_scan = self.get_objects_in_scan()

        self.nouns = self.get_nouns()
        self.nouns_sorted = self.get_nouns_sorted()

        self.label_id_mapping = self.get_label_id_mapping()

        self.graph_adj_list = self.get_graph_adj_list()

    def load_scene_graph(self):
        scene_graph = {}
        scene_graph_path = os.path.join(os.path.dirname(__file__), 'data', 'scene_graphs', self.scene_id + '.json')
        with open(scene_graph_path, 'r') as f:
            scene_graph = json.load(f)
        return scene_graph
    
    def get_nouns(self):
        scene_graph = self.load_scene_graph()
        nouns = []
        for obj in scene_graph['objects']:
            nouns.append(obj['name'])
        return nouns
    
    def get_nouns_sorted(self):
        nouns = self.get_nouns()
        return sorted(nouns)
    
    def get_objects_in_scan(self):
        objects_in_scan = []
        for obj in self.scene_graph['objects']:
            objects_in_scan.append(obj['object_id'])
        return objects_in_scan
    
    def get_label_id_mapping(self):
        scene_graph = self.load_scene_graph()
        label_id_mapping = {}
        for obj in scene_graph['objects']:
            label_id_mapping[obj['object_id']] = obj['name']
        return label_id_mapping
    
    def get_graph_adj_list(self):
        scene_graph = self.load_scene_graph()
        graph_adj_list = {}
        for obj in scene_graph['objects']:
            obj_id = obj['object_id']
            graph_adj_list[obj_id] = []
            for rel in obj['relations']:
                if rel['object_id'] in self.objects_in_scan:
                    graph_adj_list[obj_id].append(rel['object_id'])
        return graph_adj_list
    