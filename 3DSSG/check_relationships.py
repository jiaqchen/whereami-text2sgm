# Open relationships.json file

import os
import json

relationships_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/relationships.json')
relationships = {}
with open(relationships_path, 'r') as f:
    relationships = json.load(f)
    relationships = relationships['scans'] 

    print(len(relationships))
    # Find relationship with scene_id
    for r in relationships:
        if r['scan'] == 'b1f2330c-d255-2761-965e-d203c6e253c3':
            print("FOUND")
            break