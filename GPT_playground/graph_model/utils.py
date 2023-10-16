# Python suppress warnings from spaCy
import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import spacy
import en_core_web_lg
# nlp = spacy.load("en_core_web_md")
nlp = spacy.load("en_core_web_lg")

import numpy as np
import torch
import re
import json
import math

################################ DATASET LOAD ################################

def load_text_dataset(filename):
    if filename == "scanscribe_1.json":
        with open("../scripts/hugging_face/" + filename, "r") as f:
            scanscribe = json.load(f)
        
        scan_ids = set()
        dict_of_texts = scanscribe
    elif filename == "scanscribe.json":
        # open scanscribe.json
        with open("../scripts/hugging_face/" + filename, "r") as f:
            scanscribe = json.load(f)
        # load text data
        scan_ids = set()
        dict_of_texts = {}
        for s in scanscribe:
            scan_id = s['scan_id']
            scan_ids.add(scan_id)
            if scan_id not in dict_of_texts:
                dict_of_texts[scan_id] = []
            dict_of_texts[scan_id].append(s['sentence'])
    else:
        print("Invalid filename")
        return
    return scan_ids, dict_of_texts

################################ GENERAL UTILS ################################
def txt_to_json(text):
    # remove \n and change \" to "
    text = text.replace('\n', '')
    text = text.replace('\"', '"')
    # use regex to change extra spaces to be 1 space
    text = re.sub(' +', ' ', text)

    # print(text)
    # string to json
    json_data = json.loads(text)

    return json_data

    # # print(json_data)
    # # write json to file
    # with open(file_path, 'w') as outfile:
    #     json.dump(json_data, outfile, indent=4)

################################ SPACY UTILS ################################

def noun_in_list_of_nouns(noun, nouns, threshold=0.5):
    # Find the noun in nouns with the highest similarity, spacy similarity
    max_sim = 0
    max_sim_noun = None
    for n in nouns:
        # oun_vec = nlp(n)[0].vector
        sim = nlp(noun).similarity(nlp(n))
        if sim > max_sim:
            max_sim = sim
            max_sim_noun = n
    return max_sim_noun, max_sim > threshold

def vectorize_word(word):
    if word == "":
        return np.zeros(300) # TODO: hard coded because spacy vectorize is 300
    return nlp(word)[0].vector

# Recover the word given a vector
def recover_word(vector, top_n=3):
    assert(len(vector) == 300)
    ms = nlp.vocab.vectors.most_similar(
        np.asarray([vector]), n=top_n
    )
    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    return words

def print_closest_words(out, x, first_n=5):
    assert(out.shape == x.shape)
    # out and x must be n x 300 dimentional and len(shape) = 2
    if len(out.shape) != 2:
        # reshape
        out = out.reshape(-1, 300)
        x = x.reshape(-1, 300)
    # assert(out.shape[1] == 300 or out.shape[0] == 300) # TODO: hard coded
    # if len(out.shape) == 1: # 1 dimentional, so only 1 word
    #     x_word = recover_word(x)
    #     out_word = recover_word(out)
    #     print("Closest words to " + str(x_word) + ": " + str(out_word))
    #     return
    for i in range(min(out.shape[0], first_n)):
        x_word = recover_word(x[i])
        out_word = recover_word(out[i])
        print("Closest words to " + str(x_word) + ": " + str(out_word))

def print_word_distances(word1, word2):
    word1_vec = nlp(word1)[0].vector
    word2_vec = nlp(word2)[0].vector
    print("Distance between " + word1 + " and " + word2 + ": " + str(np.linalg.norm(word1_vec - word2_vec)))

def print_word_similarity(word1, word2):
    print("Similarity between " + word1 + " and " + word2 + ": " + str(nlp(word1).similarity(nlp(word2))))

################################ TORCH GRAPH UTILS ################################

def make_cross_graph(x_1_dim, x_2_dim):
    x_1_dim = x_1_dim[0]
    x_2_dim = x_2_dim[0]

    edge_index_cross = torch.tensor([[], []], dtype=torch.long)
    edge_attr_cross = torch.tensor([], dtype=torch.float)

    # Add edge from each node in x_1 to x_2
    for i in range(x_1_dim):
        for j in range(x_2_dim):
            edge_index_cross = torch.cat((edge_index_cross, torch.tensor([[i], [x_1_dim + j]], dtype=torch.long)), dim=1)
            # Add edge_attr which is dimension 1x300, all 0
            edge_attr_cross = torch.cat((edge_attr_cross, torch.zeros((1, 300), dtype=torch.float)), dim=0) # TODO: dimension 300

    assert(edge_index_cross.shape[1] == x_1_dim * x_2_dim)
    assert(edge_attr_cross.shape[0] == x_1_dim * x_2_dim)
    return edge_index_cross, edge_attr_cross

def mask_node(x, p=0.1):
    if p == 0:
        return x, None
    # Mask a random row in x with 1's
    x_clone = x.clone()
    # floor function
    num_nodes_to_mask = math.floor(x.shape[0] * p)
    if num_nodes_to_mask == 0:
        return x, None
    # num_nodes_to_mask = int(x.shape[0] * p)
    rows_to_mask = torch.randperm(x.shape[0])[:num_nodes_to_mask]
    x_clone[rows_to_mask] = 0
    return x_clone, rows_to_mask

def accuracy_score(y_pred, y_true, top_n=3, thresh=0.8):
    assert(y_pred.shape[0] == y_true.shape[0])
    # y_prd and y_true must be n x 300 dimentional, otherwise reshape
    if len(y_pred.shape) != 2:
        y_pred = y_pred.reshape(-1, 300)
        y_true = y_true.reshape(-1, 300)
    
    count_correct = 0
    for i in range(y_pred.shape[0]):
        y_pred_word = recover_word(y_pred[i], top_n=top_n) # vector of top n words
        y_true_word = recover_word(y_true[i], top_n=top_n)

        # Check if y_true_word[0] is in y_pred_word
        if y_true_word[0] in y_pred_word:
            count_correct += 1
        else: # Check if it's within some threshold of similarity
            for y in y_pred_word:
                if nlp(y_true_word[0]).similarity(nlp(y)) >= thresh:
                    count_correct += 1
                    break

    return count_correct / y_pred.shape[0]

# main
if __name__ == '__main__':
    # check the similarity between chair and couch
    things = ['shower', 'sink', 'window', 'floor', 'wall', 'mirror']
    avg_t = []
    for t in things:
        t_vector = nlp(t)[0].vector
        avg_t.append(t_vector)
    avg_t = np.mean(avg_t, axis=0)
    t_word_top_n = recover_word(avg_t)
    print("Closest to list things, ", t_word_top_n)
    print_word_similarity('bathroom', t_word_top_n[0])

