# Class for scene graph loader
import os
import json
import numpy as np
import pandas as pd

from _3dssg_utils import noun_in_list_of_nouns

class SceneGraph:
    def __init__(self, scene_id, label_mapping, euc_dist_thres=1.0):
        self.scene_id = scene_id
        self.label_mapping = label_mapping
        self.label_count_mapping = {}
        self.euc_dist_thres = euc_dist_thres # For adding edges to the graph

        self.objects_in_scan = self.get_objects_in_scan(scene_id)
        self.label_id_mapping = self.get_label_id_mapping()
        self.nouns = self.get_nouns()

        self.graph_adj_list = self.get_graph_adj_list(scene_id)

    def get_objects_in_scan(self, scene_id):
        # Path for all objects
        objects_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/objects.json')
        objects = {}
        with open(objects_path, 'r') as f:
            objects = json.load(f)
        scans = objects['scans']

        # Path for segmentations
        segmentations_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/3RScan', scene_id, 'semseg.v2.json')
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

        # Turn into pandas
        objects_in_scan = pd.DataFrame(objects_in_scan)

        # Drop unnecessary columns
        objects_in_scan = objects_in_scan.drop(columns=['attributes', 'ply_color', 'eigen13', 'rio27', 'affordances'])

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

            # Add label_mapping to objects_in_scan based on similar label
            obj_label = obj['label']

            # Get the most similar label if it's not in list already
            if (obj_label not in self.label_mapping.label_keys):
                # Get the most similar label
                max_sim_label, sim = noun_in_list_of_nouns(obj_label, self.label_mapping.label_keys, threshold=0.7)

                # TODO: do something with sim

                if max_sim_label is None:
                    # Didn't find anything at all, probably very unique or mispelled
                    self.label_count_mapping[obj_label] = 1
                else:
                    self.label_count_mapping[obj_label] = self.label_mapping.get_label_mapping()[max_sim_label][0]['count']
            else:
                self.label_count_mapping[obj_label] = self.label_mapping.get_label_mapping()[obj_label][0]['count']

        return objects_in_scan
    
    # def noun_in_list_of_nouns(noun, nouns, threshold=0.5):
    #     # Get word2vec of noun
    #     noun_vec = nlp(noun)[0].vector

    #     # Find the noun in nouns with the highest similarity, spacy similarity
    #     max_sim = 0
    #     max_sim_noun = None
    #     for n in nouns:
    #         # oun_vec = nlp(n)[0].vector
    #         sim = nlp(noun).similarity(nlp(n))
    #         if sim > max_sim:
    #             max_sim = sim
    #             max_sim_noun = n

    #     return max_sim_noun, max_sim > threshold

    def get_nouns(self):
        # Dict of noun with count
        nouns = {}
        for obj in self.objects_in_scan:
            if obj['label'] not in nouns:
                nouns[obj['label']] = 0
            nouns[obj['label']] += 1
        return nouns
    
    def get_label_id_mapping(self):
        label_id_mapping = {}
        for obj in self.objects_in_scan:
            if obj['label'] not in label_id_mapping:
                label_id_mapping[obj['label']] = []
            label_id_mapping[obj['label']].append(obj['id'])
        return label_id_mapping
    
    def get_graph_adj_list(self, scene_id):
        # Adjacency dict with key as object id and values 'label', 'adj_to', 'graph_value'
        graph_adj_list = {}
        for obj in self.objects_in_scan:
            graph_adj_list[obj['id']] = {'label': obj['label'], 'adj_to': [], 'graph_value': 0}


        # Add edges to graph
        # Open relationships
        relationships_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/relationships.json')
        relationships = {}
        with open(relationships_path, 'r') as f:
            relationships = json.load(f)
        relationships = relationships['scans']

        # Find scan with scene_id
        for r in relationships:
            if r['scan'] == scene_id:
                relationships = r
                break

        # Add edges to graph
        for rel in relationships['relationships']:
            first_obj = rel[0]
            second_obj = rel[1]
            relation = rel[3] # relation currently not used

            # If relation does not have 'than', 'same' add to graph
            if 'than' not in relation and 'same' not in relation and 'by' not in relation:
                # Only add if object obb centroid are within self.euc_dist_thres
                distance = self.get_obj_distance(first_obj, second_obj)
                if distance < self.euc_dist_thres:
                    # Add edge to graph if not already added
                    if second_obj not in graph_adj_list[str(first_obj)]['adj_to']:
                        graph_adj_list[str(first_obj)]['adj_to'].append(second_obj)
                    if first_obj not in graph_adj_list[str(second_obj)]['adj_to']:
                        graph_adj_list[str(second_obj)]['adj_to'].append(first_obj)


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