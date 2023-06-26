# Open the file 'label_mapping.csv' in data/3DSSG
import csv
import os
import json
import pandas as pd
import numpy as np

class LabelMappingLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.label_mapping = self.load_label_mapping()
        self.label_keys = self.label_mapping.keys()

    def load_label_mapping(self):
        label_mapping = {}
        # First row is columns, read into dataframe
        with open(os.path.join(self.data_dir, 'label_mapping.csv'), 'r') as f:
            label_mapping_df = pd.read_csv(f)
        # Convert to dict without index
        label_mapping = label_mapping_df.to_dict('records')

        # Convert to dict indexed by raw_category
        label_mapping_dict = {}
        for label in label_mapping:
            if label['raw_category'] not in label_mapping_dict:
                label_mapping_dict[label['raw_category']] = []
            label_mapping_dict[label['raw_category']].append(label)

        # Check if any labels have same name
        for raw_category in label_mapping_dict:
            labels = label_mapping_dict[raw_category]
            assert(len(labels) == 1)
                       
        return label_mapping_dict
    
    def get_label_mapping(self):
        return self.label_mapping
    

if __name__ == '__main__':
    # Load label mapping
    label_mapping_loader = LabelMappingLoader('../data/3DSSG')
    label_mapping = label_mapping_loader.get_label_mapping() 
    print(label_mapping['2'])
    print(len(label_mapping))

