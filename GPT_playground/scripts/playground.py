import openai
import json
import os
from tqdm import tqdm

openai.api_key_path = "../api_key.txt"

def send_completion(text, file_path):
    # system_to_send = "Could you translate the following text into a JSON representation of a scene graph? \
    #     The nodes should be the objects, also add any necessary \"attributes\" or \"affordances\". \
    #         The edges between the nodes are the spatial relationships between the objects. Make sure \
    #             to capture those as well.\n\n"

    # system_to_send = "Return a JSON with the same format below, with information from the text below. \
    #     The nodes are the objects. The edges between the nodes are the spatial relationships between \
    #         the objects. The \"attributes\" are adjectives only. Follow the format below:\n\n{\n    \
    #         \"nodes\": [\n        {\n            \"id\": \"1\",\n            \"label\": \"\",\n            \
    #         \"attributes\": [<insert adjectives>]\n        }\n    ],\n    \"edges\": [\n        \
    #         {\n            \"source\": \"\",\n            \"target\": \"\",\n            \"relationship\": \
    #         \"\"\n        }\n    ]\n}\n\n"
    # system_to_send += "\"" + text + "\""

    system_to_send = "You will only return valid JSON with the same format below, \
        given information by the user. The nodes are objects. The edges between the \
        nodes are the spatial relationships between objects. The \"attributes\" \
        are adjectives or adverbs only. Follow the format below:\n\n{\n    \"nodes\": [\n        \
        {\n            \"id\": \"1\",\n            \"label\": \"\",\n            \
        \"attributes\": [<insert adjectives>]\n        }\n    ],\n    \"edges\": \
        [\n        {\n            \"source\": \"\",\n            \"target\": \"\",\n            \
        \"relationship\": \"\"\n        }\n    ]\n}"

    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "system", "content": system_to_send},
              {"role": "user", "content": text}],
    temperature=0.0
    )

    # Save completion to output file
    with open(file_path, 'w') as outfile:
        json.dump(completion, outfile, indent=4)


if __name__ == "__main__":
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