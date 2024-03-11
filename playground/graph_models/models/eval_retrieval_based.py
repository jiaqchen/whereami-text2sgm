import time
import argparse
import sys
import torch
import torch.cuda
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph, calculate_overlap
from model_graph2graph import BigGNN
from train_utils import k_fold, cross_entropy

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

random.seed(42)

from timing import Timer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--overlap_thr', type=float, default=0.8)
    parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_set_size', type=int, default=2847)
    parser.add_argument('--test_set_size', type=int, default=712)
    parser.add_argument('--graph_size_min', type=int, default=4, help='minimum number of nodes in a graph')
    parser.add_argument('--contrastive_loss', type=bool, default=True)
    parser.add_argument('--valid_top_k', nargs='+', type=int, default=[1, 2, 3, 5])
    parser.add_argument('--use_attributes', type=bool, default=True)
    parser.add_argument('--entire_training_set', action='store_true')

    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--eval_iter_count', type=int, default=10)
    parser.add_argument('--out_of', type=int, default=10)
    parser.add_argument('--model_name', type=str, default=None)

    parser.add_argument('--eval_only_c', action='store_true')

    parser.add_argument('--eval_entire_dataset', action='store_true')
    parser.add_argument('--heads', type=int, default=2)

    args = parser.parse_args()
    return args

args = get_args()


def save_latex(results, filename):
    with open(filename, 'w') as f:
        for k in results:
            # write as percentage and with $0.00\pm0.00$ as [k][0] and [k][1]
            f.write(f'top {k} out of {args.out_of}: ${results[k][0]*100:.2f}\\pm{results[k][1]*100:.2f}$ \n')
            

def cos_sim(emb, emb1=None):
    if emb1 is None:
        emb1 = emb
    # cosine similarity between emb and emb.T
    cos_sim = np.zeros((len(emb), len(emb1)))
    for i in range(len(emb)):
        for j in range(len(emb1)):
            cos_sim[i][j] = np.dot(emb[i], emb1[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb1[j]))
    return cos_sim


@torch.no_grad()
def eval_retrieval_based(model, database, text_graphs, dataset_name, timer):
    assert(random_text is not None and random_graph is not None)

    # set model to eval mode
    model.eval()
    # Get text_graphs sorted by scene
    scene_to_text_map = {}
    for text_graph_id in text_graphs:
        scene_id = text_graph_id.split('_')[0]
        if scene_id not in scene_to_text_map:
            scene_to_text_map[scene_id] = []
        scene_to_text_map[scene_id].append(text_graph_id)
    
    # Get the text graph ids
    text_graphs_sorted = []
    db_graph_scene_ids = list(scene_to_text_map.keys()) # unique scenes in text_graphs
    for scene_id in db_graph_scene_ids: 
        text_graphs_sorted += scene_to_text_map[scene_id] # take all text descriptions, but they are sorted by scene

    # given a text query, want to get top args.top_k matches out of args.out_of
    # do args.eval_iters iterations

    # in every iteration, calculate average accuracy and variance
    eval_iters_top_k = {k: [] for k in args.valid_top_k}
    for _ in tqdm(range(args.eval_iters)):
        # in every sub_iteration, calculate how many in or out (accuracy)
        eval_iters_count_top_k = {k: [] for k in args.valid_top_k}
        for _ in range(args.eval_iter_count):
            # Choose 1 query text
            random_scene_for_text = random.choice(db_graph_scene_ids)
            q = text_graphs[random.choice(scene_to_text_map[random_scene_for_text])]
            # encode text
            x_node_ft, x_edge_idx, x_edge_ft = q.to_pyg()
            p_node_ft, p_edge_idx, p_edge_ft = random_graph.to_pyg()
            t1 = time.time()
            x_p, _, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'), 
                                torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
            timer.text2graph_text_embedding_time.append(time.time() - t1)
            timer.text2graph_text_embedding_iter.append(1)

            # Choose args.out_of graphs, remove random_scene_for_text
            db_graph_scene_ids_copy = db_graph_scene_ids.copy()
            db_graph_scene_ids_copy.remove(random_scene_for_text)
            out_of_graphs = random.sample(db_graph_scene_ids_copy, args.out_of-1)
            out_of_graphs.append(random_scene_for_text)
            assert(len(out_of_graphs) == args.out_of)

            cos_sims = []
            # pick args.out_of out of all, calculate in or out
            for scene_id in out_of_graphs:
                db = database[scene_id]
                # encode graphs
                x_node_ft, x_edge_idx, x_edge_ft = random_text.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = db.to_pyg()
                _, p_p, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                    torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                    torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                # calculate cosine similarity
                t1 = time.time()
                cos_sims.append(np.dot(x_p.cpu().numpy(), p_p.cpu().numpy()) / (np.linalg.norm(x_p.cpu().numpy()) * np.linalg.norm(p_p.cpu().numpy())))
                timer.text2graph_text_embedding_matching_score_time.append(time.time() - t1)
                timer.text2graph_text_embedding_matching_score_iter.append(1)
            
            # sort cos_sims, get top args.top_k
            t1 = time.time()
            cos_argsort = np.argsort(cos_sims)
            timer.text2graph_matching_time.append(time.time() - t1)
            timer.text2graph_matching_iter.append(1)
            cos_argsort = cos_argsort[::-1] # high cos sim is match
            for k in args.valid_top_k: eval_iters_count_top_k[k].append(args.out_of-1 in cos_argsort[:k])

        # calculate average accuracy and variance
        for k in args.valid_top_k: eval_iters_top_k[k].append(np.mean(eval_iters_count_top_k[k]))

    for k in args.valid_top_k: eval_iters_top_k[k] = (np.mean(eval_iters_top_k[k]), np.var(eval_iters_top_k[k]))
    # print results
    for k in args.valid_top_k:
        print(f'Average accuracy for top {k} matches: {eval_iters_top_k[k]}')
    
    timer.save(f'./retrieval_based_{dataset_name}_timer.txt', args)
    return eval_iters_top_k

if __name__ == "__main__":

    # 3DSSG
    # _3dssg_graphs = torch.load('../data_checkpoints/processed_data/training/3dssg_graphs_train_graph_min_size_4.pt')                # Len 1323   
    _3dssg_graphs = {}
    _3dssg_scenes = torch.load('../data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid, 
                                            graph_type='3dssg', 
                                            graph=_3dssg_scenes[sceneid], 
                                            max_dist=1.0, embedding_type='word2vec',
                                            use_attributes=args.use_attributes)     
        
    # scanscribe train
    # scanscribe_graphs = torch.load('../data_checkpoints/processed_data/training/scanscribe_graphs_train_graph_min_size_4.pt')       # 80% split len 2847
    scanscribe_graphs = {}
    scanscribe_scenes = torch.load('../data_checkpoints/processed_data/training/scanscribe_graphs_train_final_no_graph_min.pt')
    for scene_id in tqdm(scanscribe_scenes):
        txtids = scanscribe_scenes[scene_id].keys()
        assert(len(set(txtids)) == len(txtids)) # no duplicate txtids
        assert(len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1))) # no missing txtids
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe', 
                                                                        graph=scanscribe_scenes[scene_id][txt_id], 
                                                                        embedding_type='word2vec',
                                                                        use_attributes=args.use_attributes)

    # preprocess so that the graphs all at least 1 edge
    print(f'number of scanscribe graphs before removing graphs with 1 edge: {len(scanscribe_graphs)}')
    to_remove = []
    for g in scanscribe_graphs:
        if len(scanscribe_graphs[g].edge_idx[0]) <= 1: # TODO: turn into strict inequality
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs[g]
    print(f'number of scanscribe graphs after removing graphs with 1 edge: {len(scanscribe_graphs)}')

    # we use a random graph as the graph side input to get all text embeddings
    text_graph_keys = list(set([k.split('_')[0] for k in scanscribe_graphs]))
    random_graph_key = random.choice(text_graph_keys)
    random_graph = _3dssg_graphs[random_graph_key]
    _3dssg_graphs.pop(random_graph_key)
    # we use a random text as the text side input to get all graph embeddings
    random_text = None
    copy_text_graphs = scanscribe_graphs.copy()
    for text_graph_key in scanscribe_graphs:
        if text_graph_key.split('_')[0] == random_graph_key:
            random_text = scanscribe_graphs[text_graph_key]
            print('found one text from the random graph scene')
            copy_text_graphs.pop(text_graph_key)
    assert(random_text is not None and random_graph is not None)
    scanscribe_graphs = copy_text_graphs # does not include random graph texts

    # ScanScribe Test
    # scanscribe_graphs_test = torch.load('../data_checkpoints/processed_data/testing/scanscribe_graphs_test_graph_min_size_4.pt')    # 20% split len 712
    scanscribe_graphs_test = {}
    scanscribe_scenes = torch.load('../data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt')
    for scene_id in tqdm(scanscribe_scenes):
        txtids = scanscribe_scenes[scene_id].keys()
        assert(len(set(txtids)) == len(txtids)) # no duplicate txtids
        assert(len(set(txtids)) == len(range(max([int(id) for id in txtids]) + 1))) # no missing txtids
        for txt_id in txtids:
            txt_id_padded = str(txt_id).zfill(5)
            scanscribe_graphs_test[scene_id + '_' + txt_id_padded] = SceneGraph(scene_id,
                                                                        txt_id=txt_id,
                                                                        graph_type='scanscribe', 
                                                                        graph=scanscribe_scenes[scene_id][txt_id], 
                                                                        embedding_type='word2vec',
                                                                        use_attributes=args.use_attributes)

            
    print(f'number of scanscribe test graphs before removing: {len(scanscribe_graphs_test)}')
    to_remove = []
    for g in scanscribe_graphs_test:
        if len(scanscribe_graphs_test[g].edge_idx[0]) < 1:
            to_remove.append(g)
    for g in to_remove: del scanscribe_graphs_test[g]
    print(f'number of scanscribe test graphs after removing: {len(scanscribe_graphs_test)}')

    # Human Test
    # human_graphs_test = torch.load('../data_checkpoints/processed_data/testing/human_graphs_test_graph_min_size_4.pt')              # Len 35
    h_graphs_test = torch.load('../data_checkpoints/processed_data/human/human_graphs_processed.pt')
    h_graphs_remove = [k for k in h_graphs_test if k.split('_')[0] not in _3dssg_graphs]
    print(f'to remove human_graphs, hopefully none: {h_graphs_remove}')
    for k in h_graphs_remove: del h_graphs_test[k]
    assert(all([k.split('_')[0] in _3dssg_graphs for k in h_graphs_test]))
    human_graphs_test = {k: SceneGraph(k.split('_')[0], 
                                   graph_type='human',
                                   graph=h_graphs_test[k],
                                   embedding_type='word2vec',
                                   use_attributes=args.use_attributes) for k in h_graphs_test}
    
    model_name = args.model_name
    model_state_dict = torch.load(f'../model_checkpoints/graph2graph/{model_name}.pt')
    model = BigGNN(args.N, args.heads).to('cuda')
    model.load_state_dict(model_state_dict)
    

    scanscribe_timer = Timer()
    if args.eval_entire_dataset:
        args.valid_top_k = [1, 5, 10, 20, 30]
        args.out_of = len(list(set([k.split('_')[0] for k in scanscribe_graphs_test])))
        assert(args.out_of == 55)
    # ScanScribe
    scanscribe_results = eval_retrieval_based(model, _3dssg_graphs, scanscribe_graphs_test, 'scanscribe', scanscribe_timer)
    save_latex(scanscribe_results, f'./retrieval_based_scanscribe_results_topkentiredataset.txt')

    human_timer = Timer()
    if args.eval_entire_dataset:
        args.valid_top_k = [1, 5, 10, 20, 30, 50, 75]
        args.out_of = len(list(set([k.split('_')[0] for k in human_graphs_test])))
        assert(args.out_of == 140)
    # Human
    human_results = eval_retrieval_based(model, _3dssg_graphs, human_graphs_test, 'human', human_timer)
    save_latex(human_results, f'./retrieval_based_human_results_topkentiredataset.txt')