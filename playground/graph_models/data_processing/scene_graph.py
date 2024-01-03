import json
import torch
from tqdm import tqdm

from scene_graph_utils import check_valid_graph

class Node:
    def __init__(self, idx, label_features, attribute_features):
        self.idx = idx
        self.label_features = label_features
        self.attribute_features = attribute_features
        assert(type(self.attribute_features) == list)
        self.features = None
    
class Edge:
    def __init__(self, from_idx, to_idx, features):
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.features = features

class SceneGraph:
    def __init__(self, graph_type, graph, max_dist, embedding_type='ada'):
        if graph_type == '3dssg':
            self.nodes = self.extract_nodes(graph['objects'], embedding_type)
            self.edge_idx, _, self.edge_features = self.extract_edges(graph['edge_lists'], max_dist, embedding_type)
            assert(len(self.edge_idx[0]) == len(self.edge_features))
            assert(check_valid_graph(self.nodes, self.edge_idx))
        elif graph_type == 'scanscribe':
            self.nodes = []
            self.edge_idx = []

    def extract_nodes(self, objects, embedding_type='ada'):
        nodes = {}
        for objid in objects:
            obj = objects[objid]
            attributes_list = [a for attr in obj['attributes_' + embedding_type] for a in obj['attributes_' + embedding_type][attr]]
            if len(attributes_list): assert(len(attributes_list[0]) == len(obj['label_' + embedding_type]))
            node = Node(obj['id'], obj['label_' + embedding_type], attributes_list)
            nodes[obj['id']] = node
        return nodes

    def extract_edges(self, edge_lists, max_dist, embedding_type='ada'):
        edge_idx = []
        from_edge = []
        to_edge = []
        edge_attributes = []
        edge_attributes_embedding = []
        for idx, d in enumerate(edge_lists['distance']):
            if d <= max_dist:
                from_edge.append(edge_lists['from'][idx])
                to_edge.append(edge_lists['to'][idx])
                edge_attributes.append(edge_lists['relation'][idx])
                edge_attributes_embedding.append(edge_lists['relation_' + embedding_type][idx])
        assert(len(from_edge) == len(to_edge))
        edge_idx.append(from_edge)
        edge_idx.append(to_edge)            
        return edge_idx, edge_attributes, edge_attributes_embedding

    def get_subgraph(self, node_ids):
        subgraph_nodes = []
        subgraph_edge_ids_from = []
        subgraph_edge_ids_to = []
        for node in self.nodes:
            if node.idx in node_ids:
                subgraph_nodes.append(node.features)
        for from_idx, to_idx in zip(self.edge_idx[0], self.edge_idx[1]):
            if from_idx in node_ids and to_idx in node_ids:
                subgraph_edge_ids_from.append(from_idx)
                subgraph_edge_ids_to.append(to_idx)
        subgraph_edge_ids = []
        subgraph_edge_ids.append(subgraph_edge_ids_from)
        subgraph_edge_ids.append(subgraph_edge_ids_to)
        return subgraph_nodes, subgraph_edge_ids, None # node_features, edge_idx, edge_feature
    
    def to_pyg(self):
        node_ids = [int(node.idx) for node in self.nodes]
        edge_ids = self.edge_idx

        nodeid_map = {}
        for idx, nodeid in enumerate(node_ids):
            nodeid_map[nodeid] = idx
        
        edge_ids_remap = []
        edge_ids_from = []
        edge_ids_to = []
        for from_idx, to_idx in zip(edge_ids[0], edge_ids[1]):
            edge_ids_from.append(nodeid_map[from_idx])
            edge_ids_to.append(nodeid_map[to_idx])
        edge_ids_remap.append(edge_ids_from)
        edge_ids_remap.append(edge_ids_to)

        node_features = [node.features for node in self.nodes]
        return node_features, edge_ids_remap, None # node_features, edge_idx, edge_features


if __name__ == '__main__':
    ######## 3DSSG #########
    _3dssg_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')
    for sceneid in tqdm(_3dssg_scenes):
        sg = SceneGraph('3dssg', _3dssg_scenes[sceneid], max_dist=1.0, embedding_type='ada')
    
    ######### ScanScribe #########
    # scanscribe_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/raw_data/scanscribe/scanscribe_cleaned_original.pt')
    # random id
    # import random
    # random.seed(3)
    # scene_id = random.choice(list(scanscribe_scenes.keys()))

    # for scene_id in tqdm(scanscribe_scenes):
        # txtids = scanscribe_scenes[scene_id].keys()
        # assert(len(set(txtids)) == len(txtids)) # no duplicate txtids
        # assert(len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1))) # no missing txtids
