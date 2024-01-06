import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import random
import matplotlib.pyplot as plt

sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('/home/julia/Documents/h_coarse_loc/')
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph, calculate_overlap
from model_graph2graph import BigGNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

def train_graph2graph(_3dssg_graphs, scanscribe_graphs):
    model = BigGNN(args.N).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    iter = 0
    graph_size_min = 5

    # sample first 100 in scanscribe
    scanscribe_graphs = {k: scanscribe_graphs[k] for k in random.sample(list(scanscribe_graphs), 30)}

    for epoch in tqdm(range(args.epoch)):
        for scribe_id in scanscribe_graphs:
            scribe_g = scanscribe_graphs[scribe_id]
            _3dssg_g = _3dssg_graphs[scribe_id.split('_')[0]]
            if (len(scribe_g.nodes) < graph_size_min) or (len(_3dssg_g.nodes) < graph_size_min): continue

            iter += 1

            # Get negative sample until overlap is less than args.overlap_thr
            # overlap_n, overlap_iter = 1.0, 0
            # while (overlap_n > args.overlap_thr and overlap_iter < 1000):
                # _3dssg_g_n = _3dssg_graphs[np.random.choice(list(_3dssg_graphs.keys() - scribe_id.split('_')[0]))]
                # scribe_g_subgraph_n, _3dssg_g_subgraph_n = get_matching_subgraph(scribe_g, _3dssg_g_n)
                # overlap_n = calculate_overlap(scribe_g_subgraph_n, _3dssg_g_subgraph_n, args.cos_sim_thr)
                # overlap_iter += 1 # just take a random one if no good overlap found after 10000 overlap_iter

            scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g) # TODO: subgraphing cannot be so absolute, must be able to have some nodes...
            overlap = calculate_overlap(scribe_g_subgraph, _3dssg_g_subgraph, args.cos_sim_thr)
            if overlap < args.overlap_thr: print(f'Warning: positive pair overlap is less than threshold: {overlap}')

            # x = torch.tensor([scribe_g.nodes[i].features for i in scribe_g.nodes]).to('cuda') # TODO: Why is x not the same as x_node_ft?
            # p = torch.tensor([_3dssg_g.nodes[i].features for i in _3dssg_g.nodes]).to('cuda')
            # n = torch.tensor([_3dssg_g_n.nodes[i].features for i in _3dssg_g_n.nodes]).to('cuda')

            x_node_ft, x_edge_idx, x_edge_ft = scribe_g_subgraph.to_pyg() # scribe_g.to_pyg()
            p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg() # _3dssg_g.to_pyg()
            # n_node_ft, n_edge_idx, n_edge_ft = _3dssg_g.to_pyg()
            if len(x_edge_idx[0]) <= 2 or len(p_edge_idx[0]) <= 2: continue # TODO: make sure subgraphs always have something

            x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft)).to('cuda'), torch.tensor(np.array(p_node_ft)).to('cuda'),
                                    torch.tensor(x_edge_idx).to('cuda'), torch.tensor(p_edge_idx).to('cuda'),
                                    torch.tensor(np.array(x_edge_ft)).to('cuda'), torch.tensor(np.array(p_edge_ft)).to('cuda'))
            # x_n, n_n, m_n = model(torch.tensor(np.array(x_node_ft)).to('cuda'), torch.tensor(np.array(n_node_ft)).to('cuda'),
                                    # torch.tensor(x_edge_idx).to('cuda'), torch.tensor(n_edge_idx).to('cuda'),
                                    # torch.tensor(np.array(x_edge_ft)).to('cuda'), torch.tensor(np.array(n_edge_ft)).to('cuda'))
            
            loss1 = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
            # loss2 = 1 - F.cosine_similarity(x_n, n_n, dim=0) # [0, 2] 2 is good
            # loss3 = (1 - m_p) + m_n

            loss = loss1 + (1 - m_p)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                wandb.log({"loss1": loss1.sum().item(),
                            # "loss2": loss2.sum().item(),
                            # "loss3": loss3.sum().item(),
                            "loss": loss.item(),
                            "match_prob_pos": m_p.item()})
                            # "match_prob_neg": m_n.item()})
        
        wandb.log({"loss_per_epoch": loss.item()})
    return model


def evaluate_model(model, val_3dssg, val_scribe):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--overlap_thr', type=float, default=0.8)
    parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    args = parser.parse_args()

    wandb.config = { "architecture": "self attention cross attention",
                "dataset": "ScanScribe_cleaned"} # ScanScribe_1 is the cleaned dataset with ada_002 embeddings
    for arg in vars(args): wandb.config[arg] = getattr(args, arg)
    wandb.init(project="graph2graph",
                mode=args.mode,
                config=wandb.config)

    ######## 3DSSG ######### 1335 3DSSG graphs
    _3dssg_graphs = {}
    _3dssg_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')
    for sceneid in tqdm(_3dssg_scenes):
        _3dssg_graphs[sceneid] = SceneGraph(sceneid, 
                                            graph_type='3dssg', 
                                            graph=_3dssg_scenes[sceneid], 
                                            max_dist=1.0, embedding_type='word2vec')

    ######### ScanScribe ######### 218 ScanScribe scenes, more graphs
    scanscribe_graphs = {}
    scanscribe_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/scanscribe/scanscribe_cleaned_original_node_edge_features.pt')
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
                                                                        embedding_type='word2vec')
            
    model = train_graph2graph(_3dssg_graphs, scanscribe_graphs)
    val_3dssg = None
    val_scribe = None
    evaluate_model(model, val_3dssg, val_scribe)