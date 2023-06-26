# Converts text files to json files

import os
import json
import argparse

def convert_text_to_json(scene_id):
    # Read text file
    text_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/3RScan_descriptions', scene_id, 'description.txt')

    with open(text_path, 'r') as f:
        text = f.read()

    # Text is split by new line, turn into json
    text = text.split('\n')
    text = [t for t in text if t != '']

    # Create json as a list with each element as a dict with 'description_id' and 'description'
    json_list = []
    for i, t in enumerate(text):
        json_list.append({'description_id': i, 'description': t})

    # Save json with indents
    json_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/3RScan_descriptions', scene_id, 'description.json')
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent=4)

    return

# main
if __name__ == '__main__':
    # Read arguments for scene_id
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True, help='Scene ID of the scan')
    args = parser.parse_args()

    scene_id = args.scene_id
    convert_text_to_json(scene_id)