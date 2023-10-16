# Class for scene graph loader
from typing import List
import os
import json
import numpy as np
import pandas as pd

from utils import noun_in_list_of_nouns, vectorize_word, txt_to_json
from create_text_embeddings import create_embedding

graph_origin_type = ['3DSSG', 'human+GPT']

# Node configs
node_configs = {'node_label_vec_dim': 300, # spaCy word2vec dim, only allows 300
                'node_attributes_vec_dim': 300,
                'edge_features_vec_dim': 300}

embedding_map = {}

class Node:
    def __init__(self, node_type, obj_id, label, attributes):
        assert(type(obj_id) == int)
        self.node_type = node_type          # str
        self.obj_id = obj_id # TODO: make sure this is unique
        self.label = label
        self.attributes = attributes
        self.features = self.set_features_ada_002_embedding(label, attributes)
        # if label in embedding_map:
        #     self.label = embedding_map[label]
        # else:
        #     self.label = create_embedding(label)
        #     assert(len(self.label) == 1536)
        #     embedding_map[label] = self.label

        # if (node_type == 'place'):
        #     self.label = label
        #     self.attributes = None
        #     # self.features = np.zeros(node_configs['node_label_vec_dim']) # TODO: dimension of node features should be generalizable
        #     self.features = self.set_features(self.label, attributes) # TODO: place node should have "place" as label and appropriate feature
        # else:
        #     assert(type(label) == str)
        #     self.label = self.clean_label(label)    # str
        #     self.attributes = attributes            # list of str
        #     self.features = self.set_features(self.label, attributes)   # np.array

    def set_features_ada_002_embedding(self, label: str, attributes: List[str]):
        # Turn attributes into string with spaces
        if attributes is None:
            attributes = ''
        else:
            attributes = ' '.join(attributes)

        # Concatenate label and attributes
        text = label + ' ' + attributes
        if text in embedding_map:
            feature = embedding_map[text]
        else:
            feature = create_embedding(text)
            embedding_map[text] = feature
        assert(len(feature) == 1536)
        return feature

    def clean_label(self, label):
        label = label.lower()
        label = label.split(' ')
        label = label[-1]
        return label

    def set_features(self, label, attributes):
        label = self.vectorize_label(label)
        attributes = self.vectorize_attributes(attributes)
        assert(len(label) == len(attributes))
        # features = np.concatenate((label, attributes), axis=0)
        features = label + attributes
        assert(len(features) == 300) # TODO: hard coded
        # return features
        return features # TODO: Only use label for now, dim = 300
    
    def vectorize_label(self, label):
        return vectorize_word(label)

    def vectorize_attributes(self, attributes):
        # Need to handle if attributes is a JSON dict or a list
        if (attributes is None) or (len(attributes) == 0):
            return np.zeros(node_configs['node_attributes_vec_dim'])
        if type(attributes) == dict:
            values = list(attributes.values())
            values = [item for sublist in values for item in sublist]
            values = np.array([vectorize_word(v) for v in values])
        elif type(attributes) == list:
            values = np.array([vectorize_word(v) for v in attributes])
            
        return np.mean(values, axis=0)

class Edge:
    def __init__(self, source, target, relation):
        assert(type(source) == int)
        assert(type(target) == int)
        assert(type(relation) == str)
        self.source = source
        self.target = target
        if relation != '':
            self.relation = self.vectorize_relation(relation)
        else:
            self.relation = np.zeros(node_configs['edge_features_vec_dim'])
        assert(len(self.relation) == node_configs['edge_features_vec_dim'])

    def vectorize_relation(self, relation):
        return vectorize_word(relation)

class GraphLoader:
    # optional parameters euc_dist_thres and raw_json
    def __init__(self, graph_type, scene_id, euc_dist_thres=1.0, raw_json=None):
        assert(graph_type in graph_origin_type)
        if (graph_type == 'human+GPT'):
            assert(raw_json is not None)
            self.type = graph_type
            self.scene_id = scene_id
            # Access raw_json file and load into json
            with open(raw_json, 'r') as f:
                raw_json = f.read()
            graph_dict = json.loads(raw_json)
            if type(graph_dict) is str: # Still a string and need to be turned into dict
                graph_dict = txt_to_json(graph_dict) # TODO: Only needed for ScanScribe, but doesn't hurt for human+GPT, need to check though

            self.nodes = self.set_nodes(graph_type, graph_dict['nodes'])
            self.edges = self.set_edges(graph_type, graph_dict['edges'])

        elif (graph_type == '3DSSG'):
            assert(euc_dist_thres is not None)
            self.type = graph_type
            self.scene_id = scene_id
            self.euc_dist_thres = euc_dist_thres

            self.objects_in_scan = self.get_objects_in_scan(scene_id)
            self.graph_adj_list = self.get_graph_adj_list(scene_id)

            self.nodes = self.set_nodes(graph_type, self.objects_in_scan)
            self.edges = self.set_edges(graph_type, self.graph_adj_list)
        
        else:
            raise Exception('GraphLoader type not supported, or raw_json wrong')


    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        return self.edges
    
    def set_nodes(self, graph_type, objs):
        # If condition for graph type
        nodes = []
        if graph_type == '3DSSG':
            for obj in objs:
                # if (obj['label'] == 'ceiling' or obj['label'] == 'wall' or obj['label'] == 'floor'):
                #     continue
                if ('attributes' in obj.keys() and len(obj['attributes']) > 0):
                    node = Node('3dssg_node', int(obj['id'])-1, obj['label'], obj['attributes']) # TODO: -1 to make nodes index from 0
                    nodes.append(node)
                else: # TODO: No attributes, add empty attribute, check if the attribute is not used elsewhere
                    node = Node('3dssg_node', int(obj['id'])-1, obj['label'], None) # TODO: -1 to make nodes index from 0
                    nodes.append(node)
        elif graph_type == 'human+GPT': # TODO: refactor this, same procedure between 3DSSG and human+GPT
            for obj in objs:
                # if (obj['label'] == 'ceiling' or obj['label'] == 'wall' or obj['label'] == 'floor'):
                #     continue
                if ('attributes' in obj.keys() and len(obj['attributes']) > 0):
                    node = Node('human_node', int(obj['id'])-1, obj['label'], obj['attributes']) # TODO: -1 to make nodes index from 0
                    nodes.append(node)
                else:
                    node = Node('human_node', int(obj['id'])-1, obj['label'], None)
                    nodes.append(node)
        return nodes
    
    def set_edges(self, graph_type, graph_edges):
        edges = []
        if graph_type == '3DSSG':
            for obj_id, obj in graph_edges.items():
                # if (obj['label'] == 'ceiling' or obj['label'] == 'wall' or obj['label'] == 'floor'):
                #     continue
                for adj_to in obj['adj_to']:
                    edge = Edge(int(obj_id)-1, int(adj_to['id'])-1, adj_to['relation']) # TODO: -1 to make nodes index from 0
                    # edge = Edge(int(obj_id), int(adj_to['id']), adj_to['relation'])
                    edges.append(edge)
        elif graph_type == 'human+GPT':
            for edge in graph_edges:
                # if (obj['label'] == 'ceiling' or obj['label'] == 'wall' or obj['label'] == 'floor'):
                #     continue
                # Check if edge is valid
                if not edge['source'].isnumeric() or not edge['target'].isnumeric():
                    print('Warning: Edge not valid ' + str(edge))
                    continue
                edge = Edge(int(edge['source'])-1, int(edge['target'])-1, edge['relationship']) # TODO: -1 to make nodes index from 0
                # edge = Edge(int(edge['source']), int(edge['target']), edge['relationship']) 
                edges.append(edge)
        return edges

    def get_objects_in_scan(self, scene_id):
        # Path for all objects
        objects_path = os.path.join(os.path.dirname(__file__), '../../data/3DSSG/objects.json')
        objects = {}
        with open(objects_path, 'r') as f:
            objects = json.load(f)
        scans = objects['scans']

        # Path for segmentations
        segmentations_path = os.path.join(os.path.dirname(__file__), '../../data/3DSSG/3RScan', scene_id, 'semseg.v2.json')
        segmentations = {}
        with open(segmentations_path, 'r') as f:
            segmentations = json.load(f)
        segmentations = segmentations['segGroups']

        # Find scan with scene_id
        scan = None
        for s in scans:
            if s['scan'] == scene_id:
                scan = s
                break

        objects_in_scan = scan['objects']

        # Check len of obj in scans same as len of segmentations
        assert len(objects_in_scan) == len(segmentations)

        objects_in_scan = pd.DataFrame(objects_in_scan)
        objects_in_scan = objects_in_scan.drop(columns=['nyu40', 'ply_color', 'eigen13', 'rio27', 'affordances', 'state_affordances', 'symmetry'], errors='ignore') # Ignore errors if column does not exist

        # Convert back to list of dicts
        objects_in_scan = objects_in_scan.to_dict('records')

        # Turn segmentation into a dict indexted by id
        segmentations_dict = {}
        for seg in segmentations:
            segmentations_dict[str(seg['id'])] = seg
        
        # Add segmentation to objects_in_scan per object
        for obj in objects_in_scan:
            # Check that we are adding the correct segmentation to the correct object based on label
            assert obj['label'] == segmentations_dict[obj['id']]['label']

            obj['obb'] = segmentations_dict[obj['id']]['obb']

        return objects_in_scan
    
    # def get_label_id_mapping(self):
    #     label_id_mapping = {}
    #     for obj in self.objects_in_scan:
    #         if obj['label'] not in label_id_mapping:
    #             label_id_mapping[obj['label']] = []
    #         label_id_mapping[obj['label']].append(obj['id'])
    #     return label_id_mapping
    
    # def get_nouns(self):
    #     # Dict of noun with count
    #     nouns = {}
    #     for obj in self.objects_in_scan:
    #         if obj['label'] not in nouns:
    #             nouns[obj['label']] = 0
    #         nouns[obj['label']] += 1
    #     return nouns
    
    def get_graph_adj_list(self, scene_id):
        # Adjacency dict with key as object id and values 'label', 'adj_to', 'graph_value'
        graph_adj_list = {}
        for obj in self.objects_in_scan:
            graph_adj_list[obj['id']] = {'label': obj['label'], 'adj_to': []}

        # Add edges to graph
        relationships_path = os.path.join(os.path.dirname(__file__), '../../data/3DSSG/relationships.json')
        relationships = {}
        with open(relationships_path, 'r') as f:
            relationships = json.load(f)
        relationships = relationships['scans']

        relationship = None
        # Find scan with scene_id
        for r in relationships:
            if r['scan'] == scene_id:
                relationship = r
                break
        
        if relationship is None:
            raise Exception('No relationship found for this scene')
        
        # Add edges to graph
        for rel in relationship['relationships']:
            first_obj = rel[0]  # int
            second_obj = rel[1] # int
            relation = rel[3]   # str; first_obj --> relation --> second_obj
                                # side table 'standing on' floor

            if second_obj not in graph_adj_list.keys():
                continue

            # If relation does not have 'than', 'same' add to graph
            if 'than' not in relation and 'same' not in relation and 'by' not in relation:
                # Only add if object obb centroid are within self.euc_dist_thres
                distance = self.get_obj_distance(first_obj, second_obj)
                if distance < self.euc_dist_thres:
                    adj_to_dict = {}
                    adj_to_dict['id'] = second_obj
                    adj_to_dict['relation'] = relation
                    # Add edge to graph if not already added
                    if second_obj not in graph_adj_list[str(first_obj)]['adj_to']:
                        graph_adj_list[str(first_obj)]['adj_to'].append(adj_to_dict)
                    else:
                        print('Warning: Edge already added')
                    # if first_obj not in graph_adj_list[str(second_obj)]['adj_to']:
                        # graph_adj_list[str(second_obj)]['adj_to'].append(first_obj)

        # print(graph_adj_list)
        # print(graph_adj_list['14']['label'], graph_adj_list['13']['label'])
        return graph_adj_list
    
    def get_obj_distance(self, first_obj, second_obj):
        first_obj_centroid = self.get_obj_centroid(first_obj)
        second_obj_centroid = self.get_obj_centroid(second_obj)
        distance = np.linalg.norm(first_obj_centroid - second_obj_centroid)
        return distance
    
    def get_obj_centroid(self, obj_id):
        obj = self.get_obj_by_id(obj_id)
        obj_centroid = np.array(obj['obb']['centroid'])
        return obj_centroid
    
    def get_obj_by_id(self, obj_id):
        for obj in self.objects_in_scan:
            if obj['id'] == str(obj_id):
                return obj
        return None

class SceneGraph:
    def __init__(self, type, scene_id, euc_dist_thres=1.0, raw_json=None):
        self.type = type
        self.scene_id = scene_id

        if self.type == '3DSSG':
            assert(euc_dist_thres is not None)
            self.graph_loader = GraphLoader(type, scene_id, euc_dist_thres=euc_dist_thres, raw_json=None)
        elif self.type == 'human+GPT':
            assert(raw_json is not None)
            self.graph_loader = GraphLoader(type, scene_id, euc_dist_thres=None, raw_json=raw_json)
        elif self.type == 'cross':
            self.nodes = None
            self.edges = None
            return
        else:
            raise Exception('SceneGraph type not supported')
  
        self.nodes = self.graph_loader.get_nodes()
        self.edges = self.graph_loader.get_edges()
    
    def get_place_node_idx(self):
        for node in self.nodes:
            if node.node_type == 'place':
                return node, node.obj_id
        return None, None
    
    def test_nodes_edges_index(self, nodes, edges):
        # Make srue all nodes index from 0, and there are no gaps
        obj_ids = [node.obj_id for node in nodes]
        assert(len(obj_ids) == len(set(obj_ids)))
        assert(min(obj_ids) == 0)
        assert(max(obj_ids) == len(obj_ids)-1)

        # Make sure all edges index from 0, and there are no gaps
        sources = [edge.source for edge in edges]
        targets = [edge.target for edge in edges]
        for s in sources:
            assert(s in obj_ids)
        for t in targets:
            assert(t in obj_ids)

    def to_pyg(self):
        # Get only node obj_id
        obj_ids = [node.obj_id for node in self.nodes]

        # Make sure all nodes index from 0, and there are no gaps
        mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(obj_ids))}

        # Remap nodes and edges
        for node in self.nodes:
            node.obj_id = mapping[node.obj_id]
        for edge in self.edges:
            edge.source = mapping[edge.source]
            edge.target = mapping[edge.target]

        # Make sure all nodes index from 0, and there are no gaps
        self.test_nodes_edges_index(self.nodes, self.edges)
        assert(len(self.nodes) == len(mapping))

    def add_place_node(self):
        # Check if place node already exists
        for node in self.nodes:
            if node.node_type == 'place':
                return
            
        # Make new Place node
        place_node = Node('place', obj_id=len(self.nodes), label='room', attributes=None)

        # Add edges from all nodes to place node
        for node in self.nodes:
            edge = Edge(node.obj_id, place_node.obj_id, 'in') # TODO: should the relation here be "child"?
            self.edges.append(edge)

        self.nodes.append(place_node)
        return

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_edges(self, edges):
        self.edges = edges

    def get_nodes(self):
        return self.nodes
    
    def get_edges(self):
        return self.edges

    def get_node_features(self):
        len_features = len(self.nodes[0].features)
        node_features = []
        for node in self.nodes:
            assert(len(node.features) == len_features)
            node_features.append(node.features)
        return np.array(node_features)
    
    def get_edge_s_t_feats(self):
        sources, targets, features = [], [], []
        for edge in self.edges:
            sources.append(edge.source)
            targets.append(edge.target)
            features.append(edge.relation)
        return sources, targets, np.array(features)

        
# main function
if __name__ == '__main__':
    scene_graph = SceneGraph('3DSSG', '0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca', euc_dist_thres=1.0)
    scene_graph_human = SceneGraph('human+GPT', '0cac75ce-8d6f-2d13-8cf1-add4e795b9b0', raw_json='../output_clean/0cac75ce-8d6f-2d13-8cf1-add4e795b9b0/1_300ms/0_gpt_clean.json')

    print(scene_graph_human.get_node_features().shape)
    print()
    sources, targets, features = scene_graph_human.get_edge_s_t_feats()
    print(len(sources), len(targets), len(features))
    print(type(sources[0]), type(targets[0]), type(features[0]))
    print(features.shape)
    # print(scene_graph_human.edges[0].attributes)