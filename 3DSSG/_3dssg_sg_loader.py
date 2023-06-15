# Class for scene graph loader
import os
import json
import numpy as np
import pandas as pd

class SceneGraph:
    def __init__(self, scene_id, euc_dist_thres=1.0):
        self.scene_id = scene_id
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
        objects_in_scan = objects_in_scan.drop(columns=['attributes', 'ply_color', 'nyu40', 'eigen13', 'rio27', 'affordances'])

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
            # TODO: Add the relationships to the graph

            # Add edge to graph
            graph_adj_list[rel['subject']]['adj_to'].append(rel['object'])
            graph_adj_list[rel['object']]['adj_to'].append(rel['subject'])
    