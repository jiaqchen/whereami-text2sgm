# Convert text to json file by cleaning up spaces and lines
import json
import re
import os
from tqdm import tqdm
import argparse

def txt_to_json(text):
    # remove \n and change \" to "
    text = text.replace("\\n", '')
    text = text.replace('\\"', '"')
    # use regex to change extra spaces to be 1 space
    text = re.sub(' +', ' ', text)
    # trim " at beginning and end
    text = text.strip('"')
    print(text)
    # string to json
    json_data = json.loads(text)

    return json_data

# main
if __name__ == "__main__":
    # with open("./txt_to_json.json", "r") as f:
    #     test_text = f.read()
    #     test_text = str(test_text)
    # json_data = txt_to_json(test_text)
    # with open("./txt_to_json_reformatted.json", "w") as f:
    #     json.dump(json_data, f, indent=4)
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()

    scan_folders = os.listdir('./' + str(args.dataset))
    scan_folders.sort()

    # Loop through folders
    for folder in tqdm(scan_folders):
        # for file in folder
        files = os.listdir('./' + str(args.dataset) + '/' + folder)
        for json_file in files:
            filepath = './' + str(args.dataset) + '/' + folder + '/' + json_file

            with open(filepath) as f:
                json_format = txt_to_json(f.read())

            filepath = filepath.replace(".json", "_reformatted.json")
            with open(filepath, 'w') as f:
                json.dump(json_format, f, indent=4)