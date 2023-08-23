# Convert text to json file by cleaning up spaces and lines
import json
import re
import os
from tqdm import tqdm

def txt_to_json(text, file_path):
    # remove \n and change \" to "
    text = text.replace('\n', '')
    text = text.replace('\"', '"')
    # use regex to change extra spaces to be 1 space
    text = re.sub(' +', ' ', text)

    # print(text)
    # string to json
    json_data = json.loads(text)

    # print(json_data)
    # write json to file
    with open(file_path, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)

# main
if __name__ == "__main__":
    # Get folders in ../output_raw
    scan_folders = os.listdir('../output_raw')
    scan_folders.sort()

    # Loop through folders
    for folder in tqdm(scan_folders):
        # Make folder in ../output_clean, if not exists
        os.makedirs('../output_clean/' + folder, exist_ok=True)

        # Get folders in folder
        gifs_folder = os.listdir('../output_raw/' + folder)
        gifs_folder.sort()

        for gifs_f in gifs_folder:
            # Make folder in ../output_clean, if not exists
            os.makedirs('../output_clean/' + folder + '/' + gifs_f, exist_ok=True)

            # Get files in folder
            files = os.listdir('../output_raw/' + folder + '/' + gifs_f)
            files.sort()

            for file in files:
                # Get file path
                file_path = '../output_raw/' + folder + '/' + gifs_f + '/' + file

                # Open file
                with open(file_path) as f:
                    # Read as json
                    text = json.load(f)
                    # Get "content"
                    text = text['choices'][0]['message']['content']
                    # if text has '```'
                    if '```' in text:
                        print('found ```, ' + file_path)
                        # Only take json between ``` ```
                        text = text.split('```')[1]
                        # Remove 'json' in beginning
                        text = text[4:]

                # Convert text to json
                # file = file.replace('raw', 'clean')
                try:
                    txt_to_json(text, '../output_clean/' + folder + '/' + gifs_f + '/' + file)
                except:
                    print('Error with ' + file_path)
                    continue