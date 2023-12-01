import openai
import json
import os
from tqdm import tqdm
import argparse
import time

openai.api_key_path = "../openai_api_key.txt"

def send_completion(text, file_path=None):
    # system_to_send = "You will only return valid JSON with the same format below, \
    #     given information by the user. The nodes are objects. The edges between the \
    #     nodes are the spatial relationships between objects. The \"attributes\" \
    #     are adjectives or adverbs only. Follow the format below:\n\n{\n    \"nodes\": [\n        \
    #     {\n            \"id\": \"1\",\n            \"label\": \"\",\n            \
    #     \"attributes\": [<insert adjectives>]\n        }\n    ],\n    \"edges\": \
    #     [\n        {\n            \"source\": \"\",\n            \"target\": \"\",\n            \
    #     \"relationship\": \"\"\n        }\n    ]\n}"

    system_to_send = "Pretend you are an expert at translating a scene description into a JSON representation of the scene where nodes are objects and edges are spatial relationships between objects."
    user_to_send = "You will only return valid JSON with the same format below, given scene information by the user. The nodes are objects. The \"label\" child must be 1 word nouns. The edges between the nodes are the spatial relationships between objects. The \"attributes\" are adjectives or adverbs only. Follow the format below:\r\n\r\n{\r\n    \"nodes\": [\r\n        \r\n        {\r\n            \"id\": \"1\",\r\n            \"label\": \"<1 word noun>\",\r\n            \"attributes\": [<insert adjectives>]\r\n        }\r\n    ],\r\n    \"edges\": \r\n        [\r\n        {\r\n            \"source\": \"\",\r\n            \"target\": \"\",\r\n            \"relationship\": \"\"\r\n        }\r\n    ]\r\n}\r\n\r\nThe scene: REPLACETHIS\r\n\r\nThe JSON response:"

    # Replace substring "REPLACETHIS" in user_to_send with text
    user_to_send = user_to_send.replace("REPLACETHIS", text)

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "system", "content": system_to_send},
              {"role": "user", "content": user_to_send}],
    temperature=0.10
    )

    if (file_path != None):
        # Save completion to output file
        with open(file_path, 'w') as outfile:
            json.dump(completion, outfile, indent=4)

    return completion

def extract_content(completion):
    # TODO: this might only work for scanscribe right now
    # Extract content from completion
    content = completion.choices[0]['message']['content']
    return content

def save_completion(completion, scan_id, scanscribe_json_gpt_folder):
    # Check if scan_id folder exists and create file size of folder + 1
    if (not os.path.exists(scanscribe_json_gpt_folder + scan_id)):
        os.mkdir(scanscribe_json_gpt_folder + scan_id)
    file_name = None
    if (os.path.exists(scanscribe_json_gpt_folder + scan_id)):
        file_size = len(os.listdir(scanscribe_json_gpt_folder + scan_id))
        file_name = str(file_size) + '.json'

    # Save completion to the output file
    if (file_name == None):
        exit()
    with open(scanscribe_json_gpt_folder + scan_id + '/' + file_name, 'w') as outfile:
        json.dump(completion, outfile)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasource', type=str, default='3dssg', help='data source, can be 3dssg, scanscribe, or human')
    parser.add_argument('--filename', type=str, default=None, help='filename of the data source, should be in a json format with a dictionary of scene_id keys to list of descriptions for that scene')
    parser.add_argument('--scanscribe_folder', type=str, default=None)
    args = parser.parse_args()
    assert(args.datasource != None)
    
    # If args.datasource is equal to scanscribe_* using regex
    if (args.datasource[0:11] == 'scanscribe_'):
        with open('./hugging_face/' + str(args.filename), 'r') as f:
            scanscribe = json.load(f)

        for key in tqdm(scanscribe):
            descriptions = scanscribe[key]
            len_descriptions = len(descriptions)
            i = 0
            while i < len_descriptions:
                description = descriptions[i]
                try:
                    completion = send_completion(description)
                except:
                    print("Error with API call " + str(key) + " and description " + str(description))
                    time.sleep(5)
                    continue
                completion = extract_content(completion)
                save_completion(completion, key, args.scanscribe_folder)
                i += 1
                time.sleep(1)
        
    elif (args.datasource == 'scanscribe'):
        # Unpack scanscribe.json
        with open('./hugging_face/' + str(args.filename), 'r') as f:
            scanscribe = json.load(f)
        
        # Iterate through scanscribe using tqdm
        i = 0
        max_i = len(scanscribe)
        while i < max_i:
            print("Currently on iteration " + str(i) + " out of " + str(max_i), end='\r')
            description = scanscribe[i]['sentence']
            if (description == '') or (description == ' ') or (description[0:9] == "I'm sorry"):
                i += 1
                continue
            scan_id = scanscribe[i]['scan_id']

            completion = None
            try:
                completion = send_completion(description)
            except:
                print("Error with API call " + scan_id + " trying again.")
                continue
            if (completion == None): # shouldn't happen tho
                print("Skipped " + scan_id)
                i += 1
                continue
            completion = extract_content(completion)


            
            i += 1

        exit()

    if (args.datasource == 'human'):
        # Open data.json file in ../data/data.json
        with open('../data/data.json') as json_file:
            # Convert to proper json by adding commas between objects
            data = json_file.read()
            data = data.replace('}\n{', '},\n{')
            data = '[' + data + ']'

            # Load as json
            json_data = json.loads(data)

        # Iterate through json_data using tqdm
        for i in tqdm(range(len(json_data))):
        # for i in range(len(json_data)):
            data_pt = json_data[i]

            # Get text from data_pt
            description = data_pt['description']
            scanId_path = data_pt['scanId']

            # Remove .gif and replace / with _
            scanId = scanId_path.replace('/', '_')
            scanId = scanId.replace('.gif', '') 

            # Get the gif name, everything after first _
            gif_name = scanId.split('_')[1:]
            gif_name = '_'.join(gif_name)

            # Remove "_X_300ms" from scanId
            scanId = scanId.split('_')[0]

            if (scanId != '4acaebce-6c10-2a2a-852f-98c6902bcc88'):
                continue

            # Make a folder in output_raw if it doesn't exist, named after scanId
            if (not os.path.exists('../output_raw/' + scanId)):
                os.mkdir('../output_raw/' + scanId)

            # Make subfolder for gif_name if it doesn't exist
            if (not os.path.exists('../output_raw/' + scanId + '/' + gif_name)):
                os.mkdir('../output_raw/' + scanId + '/' + gif_name)

            # Make a new file in the gif_name folder, incremented from number of files in folder, with '_gpt_raw.json'
            num_files = len(os.listdir('../output_raw/' + scanId + '/' + gif_name)) # increment from 0
            file_name = str(num_files) + '_gpt_raw.json'
            file_path = '../output_raw/' + scanId + '/' + gif_name + '/' + file_name

            # Send text to GPT-3
            send_completion(description, file_path)