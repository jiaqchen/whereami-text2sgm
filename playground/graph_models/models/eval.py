import time
import argparse
import sys
import torch
import torch.cuda
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import wandb
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

from train import eval_acc as eval
from train import format_to_latex

from args import get_args
args = get_args()

from timing import Timer

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', type=str, default='online')
    # parser.add_argument('--epoch', type=int, default=10)
    # parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--weight_decay', type=float, default=5e-5)
    # parser.add_argument('--N', type=int, default=1)
    # parser.add_argument('--overlap_thr', type=float, default=0.8)
    # parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    # parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--training_set_size', type=int, default=2847)
    # parser.add_argument('--test_set_size', type=int, default=712)
    # parser.add_argument('--graph_size_min', type=int, default=4, help='minimum number of nodes in a graph')
    # parser.add_argument('--contrastive_loss', type=bool, default=True)
    # parser.add_argument('--valid_top_k', nargs='+', type=int, default=[1, 3, 5])
    # parser.add_argument('--use_attributes', type=bool, default=True)
    # parser.add_argument('--training_with_cross_val', type=bool, default=True)
    # parser.add_argument('--folds', type=int, default=5)
    # parser.add_argument('--skip_k_fold', type=bool, default=False)
    # args = parser.parse_args()

    wandb.config = { "architecture": "self attention cross attention",
                     "dataset": "ScanScribe_cleaned"} # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
    for arg in vars(args): wandb.config[arg] = getattr(args, arg)
    wandb.init(project="graph2graph",
                mode=args.mode,
                config=wandb.config)

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
    


    if args.eval_entire_dataset:
        model_name = model_name + '_topkoutofentire_'
    if args.eval_only_c:
        model_name = model_name + '_eval_only_c'
    model_name = model_name + '_' + str(args.eval_iters)

    start = time.time()
    scanscribe_timer = Timer()
    scanscribe_test_acc = eval(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(scanscribe_graphs_test.values()),
                                    fold=None,
                                    mode='scanscribe_test',
                                    timer=scanscribe_timer)
    scanscribe_timer.save(f'../eval_outputs/{model_name}_scanscribe_test_time.txt', args)
    print(f'accuracy on scanscribe test set: {scanscribe_test_acc}')
    end_scanscribe = time.time()
    print(f'time for scanscribe test set: {end_scanscribe - start}')
    with open(f'../eval_outputs/{model_name}_scanscribe_test_acc.txt', 'w') as f:
        scanscribe_test_acc = format_to_latex(scanscribe_test_acc)
        f.write(f'{scanscribe_test_acc}')


    start = time.time()
    human_timer = Timer()
    human_test_acc = eval(model=model,
                                    database_3dssg=_3dssg_graphs,
                                    dataset=list(human_graphs_test.values()),
                                    fold=None,
                                    mode='human_test',
                                    valid_top_k=args.valid_top_k,
                                    timer=human_timer)
    human_timer.save(f'../eval_outputs/{model_name}_human_test_time.txt', args)
    print(f'accuracy on human test set: {human_test_acc}')
    end_human = time.time()
    print(f'time for human test set: {end_human - start}')
    with open(f'../eval_outputs/{model_name}_human_test_acc.txt', 'w') as f:
        human_test_acc = format_to_latex(human_test_acc)
        f.write(f'{human_test_acc}')