import pytest
import torch
import sys

sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
from playground.graph_models.src.utils import noun_in_list_of_nouns, vectorize_word, recover_word, print_closest_words, make_cross_graph, mask_node
from playground.graph_models.models.train_utils import k_fold

####################### SPACY UTILS #######################

def test_noun_in_list_of_nouns():
    nouns = ["cat", "dog", "mouse", "elephant", "lion"]
    noun = "lioness"

    max_sim_noun, is_in = noun_in_list_of_nouns(noun, nouns, threshold=0.5)
    assert(max_sim_noun == "lion")
    assert(is_in == True)

def test_vectorize_word():
    word = "cat"
    vector = vectorize_word(word)
    assert(len(vector) == 300)

def test_recover_word():
    # Vectorize cat
    word = "cat"
    vector = vectorize_word(word)
    # Recover cat
    recovered_word = recover_word(vector)
    assert(recovered_word[0] == "cat") # First/top ranked word is cat

####################### TORCH GRAPH UTILS #######################

def test_make_cross_graph():
    x_1_dim = torch.tensor([5, 300])
    x_2_dim = torch.tensor([3, 300])
    edge_index_cross, edge_attr_cross = make_cross_graph(x_1_dim, x_2_dim)
    assert(edge_index_cross.shape[1] == 15)
    assert(edge_attr_cross.shape[0] == 15)

def test_mask_node():
    # Make random torch tensor
    x = torch.rand((5, 300))

    # Mask nothing
    x_masked = mask_node(x, p=0)
    assert(torch.all(torch.eq(x, x_masked)))

    # Mask everything with 0's
    x_masked = mask_node(x, p=1)
    assert(torch.all(torch.eq(x_masked, torch.zeros((5, 300)))))

####################### TRAIN UTILS #######################

def test_k_fold():

    dataset = [i for i in range(20)]
    train_indices, test_indices, val_indices = k_fold(dataset, 5)
    assert(len(train_indices) == 5)
    assert(len(test_indices) == 5)
    assert(len(val_indices) == 5)
    print(f'train_indies: {train_indices}')
    print(f'test_indies: {test_indices}')
    print(f'val_indies: {val_indices}')

if __name__ == '__main__':
    test_k_fold()
