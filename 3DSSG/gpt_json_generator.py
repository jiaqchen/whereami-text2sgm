import argparse
import io
import json
import sys
import os

import numpy as np

def find_in_objects(objects, object_id):
    for object in objects:
        if object['id'] == str(object_id):
            return object['label']
    return None

# main
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='scene0000_00')
    scene_id = parser.parse_args().scene_id

    # Go through data/3DSSG/relationships.json and find the same scene_id
    # open json
    with open('../data/3DSSG/relationships.json', 'r') as f:
        relationships = json.load(f)
    relationships = relationships['scans']
    # find the same scene_id
    found_relationship = None
    for relationship in relationships:
        if relationship['scan'] == scene_id:
            # get the relationship
            found_relationship = relationship['relationships']
            break

    # Go through data/3DSSG/objects.json and find the objects in the scene_id
    # open json
    with open('../data/3DSSG/objects.json', 'r') as f:
        objects = json.load(f)
    objects = objects['scans']

    # find the same scene_id
    found_objects = None
    for object in objects:
        if object['scan'] == scene_id:
            # get the objects
            found_objects = object['objects']
            break

    print(found_objects)


    # Go through found_relationships and turn each [object1, object2, relationship] into their text description
    cleaned_relationship = []
    for relation in found_relationship:
        # if fourth element is 'same object type', delete from found_relationship
        if relation[3] == 'same object type':
            continue

        # get object1 and object2 names
        object1_name = find_in_objects(found_objects, relation[0])
        object2_name = find_in_objects(found_objects, relation[1])

        add_relation = [object1_name, object2_name, relation[3]]
    
        cleaned_relationship.append(add_relation)

    print(cleaned_relationship)
