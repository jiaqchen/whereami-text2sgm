from typing import List

import openai
import argparse
import numpy as np
import torch
import os
import tiktoken
import tqdm
from utils import load_text_dataset


with open(os.path.join(os.path.dirname(__file__), '../api_key.txt'), 'r') as f:
    api_key = f.read().strip()
openai.api_key = api_key

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_list_of_strings(list_of_strings: List[str], encoding_name: str) -> int:
    """Returns the number of tokens in a list of text strings."""
    num_tokens = 0
    for string in list_of_strings:
        num_tokens += num_tokens_from_string(string, encoding_name)
    return num_tokens

def num_tokens_from_dict(dict_of_texts: dict, encoding_name: str) -> int:
    """Returns the number of tokens in a dictionary of text strings."""
    num_tokens = 0
    for scan_id in dict_of_texts:
        num_tokens += num_tokens_from_list_of_strings(dict_of_texts[scan_id], encoding_name)
    return 
    
def check_tokens():
    scan_ids, dict_of_texts = load_text_dataset()
    num_tokens = num_tokens_from_dict(dict_of_texts, 'cl100k_base')
    print(num_tokens)
    
def create_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding

def tokenize_text(filename):
    scan_ids, dict_of_texts = load_text_dataset(filename)
    dict_of_embeddings = {}
    for scan_id in tqdm.tqdm(dict_of_texts):
        dict_of_embeddings[scan_id] = []
        for text in dict_of_texts[scan_id]:
            embedding = create_embedding(text)
            dict_of_embeddings[scan_id].append(embedding)

    return dict_of_embeddings

if __name__ == '__main__':
    ## check_tokens()
    ############################### Create embeddings for text dataset
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--filename', type=str, default=None)
    # args = parser.parse_args()

    # dict_of_embeddings = tokenize_text(args.filename)
    # torch.save(dict_of_embeddings, 'human+GPT_cleaned_text_embedding_ada_002.pt')
    ##############################################

    ############################################### Check similarity for openai embeddings
    # stringa = "blue big couch"
    # stringb = "blue chair"
    # stringc = "red couch"

    # embeddinga = create_embedding(stringa)
    # embeddingb = create_embedding(stringb)
    # embeddingc = create_embedding(stringc)

    # embeddinga = np.asarray(embeddinga)
    # embeddingb = np.asarray(embeddingb)
    # embeddingc = np.asarray(embeddingc)

    # # euc distance
    # print("Euclidean distance between a and b: ", np.linalg.norm(embeddinga - embeddingb))
    # print("Euclidean distance between a and c: ", np.linalg.norm(embeddinga - embeddingc))
    # print("Euclidean distance between b and c: ", np.linalg.norm(embeddingb - embeddingc))
    ###############################################

    ############################################### Check similarity for openai embeddings
    stringa = "window"
    stringb = "window"

    embeddinga = create_embedding(stringa)
    embeddingb = create_embedding(stringb)

    embeddinga = np.asarray(embeddinga)
    embeddingb = np.asarray(embeddingb)

    # euc distance
    print("Euclidean distance between a and b: ", np.linalg.norm(embeddinga - embeddingb))
    ###############################################
