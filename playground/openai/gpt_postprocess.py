# Go through the ./scanscribe_json_gpt folder and process the json files in the subfolders such that they are indented and formatted correctly
import os
import json
import re
import tqdm

def txt_to_json(text):
    # remove \n and change \" to "
    text = text.replace("\\n", '')
    text = text.replace('\\"', '"')
    # use regex to change extra spaces to be 1 space
    text = re.sub(' +', ' ', text)
    # trim " at beginning and end
    text = text.strip('"')
    # string to json
    json_data = json.loads(text)

    return json_data

# main
if __name__ == '__main__':
    # open ./scanscribe_json_gpt
    path = './scanscribe_json_gpt'
    for root, dirs, files in tqdm.tqdm(os.walk(path)):
        for dir in dirs:
            # make the same folder in ./scanscribe_json_gpt_reformatted if it doesn't exist
            reformatted_path = './scanscribe_json_gpt_reformatted'
            if not os.path.exists(os.path.join(reformatted_path, dir)):
                os.mkdir(os.path.join(reformatted_path, dir))
            path_dir = os.path.join(root, dir)
            for root_dir, dirs_dir, files_dir in os.walk(path_dir):
                sorted(files_dir)
                for file in files_dir:
                    path_file = os.path.join(root_dir, file)
                    # open file
                    try:
                        with open(path_file, 'r') as f:
                            json_format = txt_to_json(f.read())
                    except:
                        print("Error with file: " + str(path_file))
                    
                    # write to file
                    with open(os.path.join(reformatted_path, dir, file), 'w') as f:
                        json.dump(json_format, f, indent=4)