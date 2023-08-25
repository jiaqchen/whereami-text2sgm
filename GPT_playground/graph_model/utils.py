# Python suppress warnings from spaCy
import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import spacy
import en_core_web_lg
# nlp = spacy.load("en_core_web_md")
nlp = spacy.load("en_core_web_lg")

import numpy as np
import torch

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
    return nlp(word)[0].vector

# Recover the word given a vector
def recover_word(vector):
    assert(len(vector) == 300)
    ms = nlp.vocab.vectors.most_similar(
        np.asarray([vector]), n=3
    )
    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    return words

def print_closest_words(out, x, first_n=5):
    assert(out.shape == x.shape)
    assert(out.shape[1] == 300) # TODO: hard coded
    for i in range(min(out.shape[0], first_n)):
        x_word = recover_word(x[i])
        out_word = recover_word(out[i])
        print("Closest words to " + str(x_word) + ": " + str(out_word))

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
        return x
    # Mask a random row in x with 1's
    x_clone = x.clone()
    num_nodes_to_mask = int(x.shape[0] * p) + 1
    rows_to_mask = torch.randperm(x.shape[0])[:num_nodes_to_mask]
    x_clone[rows_to_mask] = 0
    return x_clone