import os
import json

class ScanDescriptions:
    def __init__(self, scene_id):
        self.scene_id = scene_id
        self.descriptions = self.load_descriptions()

    def load_descriptions(self):
        scan_descriptions = {}
        scan_descriptions_path = os.path.join(os.path.dirname(__file__), 'data', 'scan_descriptions', self.scene_id + '.json')
        with open(scan_descriptions_path, 'r') as f:
            scan_descriptions = json.load(f)
        return scan_descriptions