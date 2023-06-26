import os
import json

class ScanDescriptions:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.descriptions = self.load_descriptions()

    def load_descriptions(self):
        scan_descriptions = {}
        scan_descriptions_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/3RScan_descriptions', self.scene_id, 'description.json')
        with open(scan_descriptions_path, 'r') as f:
            scan_descriptions = json.load(f)
        return scan_descriptions
    


# TODO:
# 5. Try to match 1 description to 1 scene. Out of 5. Percentage likelihood of each scene.
# 6. Change the logic of the graph traversal to not do a weighting per graph, but count the score overall and compare between graphs.
#   a. Just sum all the scores for all graphs