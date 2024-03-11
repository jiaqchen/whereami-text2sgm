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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
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
    parser.add_argument('--training_with_cross_val', type=bool, default=True)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--skip_k_fold', type=bool, default=False)
    parser.add_argument('--entire_training_set', action='store_true')
    parser.add_argument('--subgraph_ablation', action='store_true')
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--eval_iter_count', type=int, default=10)
    parser.add_argument('--out_of', type=int, default=10)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--loss_ablation_m', action='store_true')
    parser.add_argument('--loss_ablation_c', action='store_true')
    parser.add_argument('--eval_only_c', action='store_true')
    parser.add_argument('--continue_training', type=int, default=0)
    parser.add_argument('--continue_training_model', type=str, default=None)

    parser.add_argument('--eval_entire_dataset', action='store_true')
    parser.add_argument('--heads', type=int, default=2)
    args = parser.parse_args()
    return args

args = get_args()


def vis_cov(cov_matrix, name):

    # Visualize the covariance matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cov_matrix, interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Covariance Matrix Heatmap, ' + name)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    # plt.xticks(range(len(cov_matrix)), range(1, len(cov_matrix) + 1))
    # plt.yticks(range(len(cov_matrix)), range(1, len(cov_matrix) + 1))
    # plt.xticks(range(len(cov_matrix)), cov_ticks, rotation=90)
    # plt.yticks(range(len(cov_matrix)), cov_ticks, rotation=0)
    # fit all the tick labels
    plt.tight_layout()
    # save plot to file in './eval_emb_var_cov'
    plt.savefig(f'./eval_emb_var_cov/cov_matrix_{name}.png')
    plt.close()

def vis_cos(cov_matrix, name, cov_ticks, cov_ticks_2=None):
    # assert all values of cov_matrix is between -1 and 1 with a small margin
    assert(np.all(cov_matrix <= 1.0001) and np.all(cov_matrix >= -1.0001))
    # if values are greater, clip to 1, or -1
    cov_matrix = np.clip(cov_matrix, -1, 1)

    # Visualize the covariance matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cov_matrix, interpolation='nearest', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Cos Sim Matrix Heatmap, ' + name)
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    # plt.xticks(range(len(cov_matrix)), range(1, len(cov_matrix) + 1))
    # plt.yticks(range(len(cov_matrix)), range(1, len(cov_matrix) + 1))
    plt.yticks(range(len(cov_ticks)), cov_ticks, rotation=0)
    if cov_ticks_2 is not None: plt.xticks(range(len(cov_ticks_2)), cov_ticks_2, rotation=90)
    else: plt.xticks(range(len(cov_ticks)), cov_ticks, rotation=90)
    # fit all the tick labels
    plt.tight_layout()
    # save plot to file in './eval_emb_var_cov'
    plt.savefig(f'./eval_emb_var_cov/cov_matrix_{name}.png')
    plt.close()


def cos_sim(emb, emb1=None):
    # # first vector
    # f = emb[0]
    # g = emb[0]
    # cos_sim_f_g = np.dot(f, g) / (np.linalg.norm(f) * np.linalg.norm(g))
    # print(f'cos_sim_f_g: {cos_sim_f_g}')
    # h = emb[1]
    # cos_sim_f_h = np.dot(f, h) / (np.linalg.norm(f) * np.linalg.norm(h))
    # print(f'cos_sim_f_h: {cos_sim_f_h}')
    # exit()

    if emb1 is None:
        emb1 = emb
    # cosine similarity between emb and emb.T
    cos_sim = np.zeros((len(emb), len(emb1)))
    for i in range(len(emb)):
        for j in range(len(emb1)):
            cos_sim[i][j] = np.dot(emb[i], emb1[j]) / (np.linalg.norm(emb[i]) * np.linalg.norm(emb1[j]))
    return cos_sim


@torch.no_grad()
def emb_graphs_texts_sep(model, database, text_graphs, dataset):
    model.eval()
    # Get text_graphs sorted by scene
    scene_to_text_map = {}
    for text_graph_id in text_graphs:
        scene_id = text_graph_id.split('_')[0]
        if scene_id not in scene_to_text_map:
            scene_to_text_map[scene_id] = []
        scene_to_text_map[scene_id].append(text_graph_id)

    db_graph_scene_ids = list(scene_to_text_map.keys())[:10]

    # Go through all the graphs
    g_db_embeddings = []
    g_t_embeddings = []
    for idx, scene_id in enumerate(db_graph_scene_ids): # invidual scene
        db = database[scene_id]
        q = text_graphs[scene_to_text_map[scene_id][0]] # first text in each scene

        # Get the embeddings
        x_node_ft, x_edge_idx, x_edge_ft = q.to_pyg()
        p_node_ft, p_edge_idx, p_edge_ft = db.to_pyg()

        x_p, p_p, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
        g_db_embeddings.append(p_p) # same scene graph
        g_t_embeddings.append(x_p)  # same scene text
    
    # assert len
    assert(len(g_db_embeddings) == len(g_t_embeddings))

    # convert db_embeddings and t_embeddings to numpy arrays
    g_db_embeddings = [emb.cpu().numpy() for emb in g_db_embeddings]
    g_t_embeddings = [emb.cpu().numpy() for emb in g_t_embeddings]

    # Calculate cos matrix
    g_db_t_cos = cos_sim(g_db_embeddings, g_t_embeddings)
    vis_cos(g_db_t_cos, 'g_db_t_cos', db_graph_scene_ids)


@torch.no_grad()
def calc_cov_graph_to_text(model, database, text_graphs, dataset_name):
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
    cov_ticks = []
    ind = 0

    db_graph_scene_ids = list(scene_to_text_map.keys())[:5] # first few scenes

    for scene_id in db_graph_scene_ids: 
        text_graphs_sorted += scene_to_text_map[scene_id][:5] # takes first few descriptions from each scene
        cov_ticks.extend([ind for _ in range(len(scene_to_text_map[scene_id][:5]))])
        ind += 1

    # Go through all the graphs
    g_db_embeddings = []
    g_t_embeddings = []
    for idx, scene_id in enumerate(db_graph_scene_ids):
        scene_id_ = scene_id.split('_')[0]
        # db = database[scene_id_]
        db = database[scene_id] # Database scene graph
        db_embeddings = []
        t_embeddings = []
        for text_scene_id in tqdm(text_graphs_sorted):
            q = text_graphs[text_scene_id] # query text graph

            # Get the embeddings
            x_node_ft, x_edge_idx, x_edge_ft = q.to_pyg()
            p_node_ft, p_edge_idx, p_edge_ft = db.to_pyg()

            x_p, p_p, _ = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                    torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                    torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
            db_embeddings.append(p_p)
            t_embeddings.append(x_p)
        
        # convert db_embeddings and t_embeddings to numpy arrays
        db_embeddings = [emb.cpu().numpy() for emb in db_embeddings] # list of embeddings for each text
        t_embeddings = [emb.cpu().numpy() for emb in t_embeddings]


        # Calculate covariance matrix between first two db_embeddings, and first and last db_embedding
        # first_db_emb = db_embeddings[0]
        # second_db_emb = db_embeddings[1]
        # last_db_emb = db_embeddings[-1]
        # first_second_cov = np.cov(np.array([first_db_emb, second_db_emb]))
        # first_last_cov = np.cov(np.array([first_db_emb, last_db_emb]))
        db_cov = np.cov(np.array(db_embeddings).T)
        assert(db_cov.shape == (len(db_embeddings[0]), len(db_embeddings[0])))
        t_cov = np.cov(np.array(t_embeddings).T)
        assert(t_cov.shape == (len(t_embeddings[0]), len(t_embeddings[0])))
        # Visualize the covariance matrix
        vis_cov(db_cov, f'db_cov_{idx}_{scene_id}')
        vis_cov(t_cov, f't_cov_{idx}_{scene_id}')

        # Calculate the cos sim
        db_cos = cos_sim(db_embeddings) # all of the same scene
        t_cos = cos_sim(t_embeddings)   # different texts

        # Visualize the cos sim
        vis_cos(db_cos, f'db_cos_{idx}_{scene_id}', cov_ticks[:len(db_embeddings)])
        vis_cos(t_cos, f't_cos_{idx}_{scene_id}', cov_ticks[:len(t_embeddings)])

        g_db_embeddings.append(db_embeddings)
        g_t_embeddings.append(t_embeddings)

    # Calculate covariacne matrix between different graphs
    g_db = [db_embeddings[0] for db_embeddings in g_db_embeddings] # first embeddings from first text for each graph
    g_t = [t for t in g_t_embeddings[0]] # all text embeddings

    print(f'len of g_db: {len(g_db)}')
    print(f'len of g_t: {len(g_t)}')

    g_db_cos = cos_sim(g_db)
    # print(f'first g_db_cos: {g_db_cos[:5]}')
    g_t_cos = cos_sim(g_t)
    g_db_t_cos = cos_sim(g_db, g_t)
    vis_cos(g_db_cos, 'g_db_cos', db_graph_scene_ids)
    vis_cos(g_t_cos, 'g_t_cos', text_graphs_sorted)
    vis_cos(g_db_t_cos, 'g_db_t_cos', db_graph_scene_ids, cov_ticks_2=text_graphs_sorted)


    # For each graph, for each text, feed it through the model, keep track of which texts are correct and which are not

    # Build covariance matrix with all output graph embeddings to see correlation between them

    pass


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
    

    # ScanScribe
    calc_cov_graph_to_text(model, _3dssg_graphs, scanscribe_graphs_test, 'scanscribe')
    # emb_graphs_texts_sep(model, _3dssg_graphs, scanscribe_graphs_test, 'scanscribe')

    # Human
    # calc_cov_graph_to_text(model, _3dssg_graphs, human_graphs_test, 'human') # There's only 1 description per scene so be careful