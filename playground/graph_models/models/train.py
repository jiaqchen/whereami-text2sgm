import argparse
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
import random
import matplotlib.pyplot as plt

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph, calculate_overlap
from model_graph2graph import BigGNN

torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())

random.seed(42)

def cross_entropy(preds, targets, reduction='none', dim=-1):
    log_softmax = torch.nn.LogSoftmax(dim=dim) 
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def train_graph2graph(_3dssg_graphs, scanscribe_graphs):
    model = BigGNN(args.N).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    iter = 0

    current_keys = list(scanscribe_graphs.keys())
    assert(all([len(scanscribe_graphs[g].nodes) >= args.graph_size_min for g in scanscribe_graphs]))

    # Contrastive Loss
    if (args.contrastive_loss):
        for epoch in tqdm(range(args.epoch)):
            random.shuffle(current_keys)
            current_keys_batched = [current_keys[i:i+args.batch_size] for i in range(0, len(current_keys) - args.batch_size, args.batch_size)]
            # print(f'len(current_keys): {len(current_keys_batched)}, num batches {int(len(current_keys) / args.batch_size)}')
            # assert(len(current_keys_batched) == int(len(current_keys) / args.batch_size)) # TODO: Check the indexing is okay here, but for now should be fine we just skip a few graphs
            assert(len(current_keys_batched[0]) == args.batch_size)
            skipped = 0
            total = 0
            for batch in current_keys_batched:
                loss1 = torch.zeros((len(batch), len(batch))).to('cuda')
                loss3 = torch.zeros((len(batch), len(batch))).to('cuda')
                for i in range(len(batch)):
                    for j in range(i, len(batch)):
                        total += 1
                        scribe_g = scanscribe_graphs[batch[i]]
                        _3dssg_g = _3dssg_graphs[batch[j].split('_')[0]]
                        scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g)
                        if _3dssg_g_subgraph is None: _3dssg_g_subgraph = _3dssg_g
                        if scribe_g_subgraph is None: scribe_g_subgraph = scribe_g # TODO: why is scribe g None now?

                        x_node_ft, x_edge_idx, x_edge_ft = scribe_g_subgraph.to_pyg()
                        p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg()
                        if len(x_edge_idx[0]) <= 2 or len(p_edge_idx[0]) <= 2: 
                            skipped += 1
                            loss1[i][j] = 1
                            loss1[j][i] = loss1[i][j]
                            loss3[i][j] = 0.5
                            loss3[j][i] = loss3[i][j]
                            continue
                        x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                                torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                                torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                        # remove from cuda to free space
                        x_node_ft, x_edge_idx, x_edge_ft = None, None, None
                        loss1[i][j] = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                        loss1[j][i] = loss1[i][j]
                        loss3[i][j] = m_p
                        loss3[j][i] = loss3[i][j]
                loss1_t = (torch.ones((len(batch), len(batch))).to('cuda') - torch.eye(len(batch)).to('cuda')) * 2
                loss3_t = torch.eye(len(batch)).to('cuda')

                # Average m_p across diagonal
                avg_mp = torch.diag(loss3).mean()
                avg_mn = (torch.sum(loss3) - torch.diag(loss3).sum()) / (len(batch) * (len(batch) - 1))
                avg_cos_sim_p = torch.diag(loss1).mean()
                avg_cos_sim_n = (torch.sum(loss1) - torch.diag(loss1).sum()) / (len(batch) * (len(batch) - 1))
                # Cross entropy
                loss1 = cross_entropy(loss1, loss1_t, reduction='mean', dim=1)
                loss3 = cross_entropy(loss3, loss3_t, reduction='mean', dim=1)
                loss = (loss1 + loss3) / 2.0

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"loss1": loss1.item(),
                            "loss3": loss3.item(),
                            "loss": loss.item(),
                            "avg_matching_pos": avg_mp.item(),
                            "avg_matching_neg": avg_mn.item(),
                            "avg_cos_sim_pos": avg_cos_sim_p.item(),
                            "avg_cos_sim_neg": avg_cos_sim_n.item()})
                
            wandb.log({"loss_per_epoch": loss.item()})
            if epoch % 10 == 0:
                evaluate_model(model, scanscribe_graphs_test, _3dssg_graphs, 'test')
                print(f'x_p first 10: {x_p[:10]}')
                print(f'p_p first 10: {p_p[:10]}')
            print(f'Skipped {skipped} graphs out of {total} because one of the subgraphs had too few edges')
    else: 
        batch_size = args.batch_size
        for epoch in tqdm(range(args.epoch)):
            curr_batch = 0
            loss = 0
            skipped = 0

            for scribe_id in scanscribe_graphs:
                scribe_g = scanscribe_graphs[scribe_id]
                _3dssg_g = _3dssg_graphs[scribe_id.split('_')[0]]
                
                # TODO: 1) Implement contrastive loss instead of triplet loss
                # TODO: 2) Finalize TEST datasets (2x)
                # TODO: 3) Run things on euler

                # # Get negative sample until overlap is less than args.overlap_thr
                # # overlap_n, overlap_iter = 1.0, 0
                # # while (overlap_n > args.overlap_thr and overlap_iter < 1000):
                # scribe_g_subgraph_n, _3dssg_g_subgraph_n = None, None
                # while (scribe_g_subgraph_n is None or _3dssg_g_subgraph_n is None):
                #     # _3dssg_g_n = _3dssg_graphs[np.random.choice(list([k for k in _3dssg_graphs if k != scribe_id.split('_')[0]]))]
                #     _3dssg_g_n = _3dssg_graphs[np.random.choice([k.split('_')[0] for k in current_keys if k.split('_')[0] != scribe_id.split('_')[0]])]
                #     scribe_g_subgraph_n, _3dssg_g_subgraph_n = get_matching_subgraph(scribe_g, _3dssg_g_n)
                #     # overlap_n = calculate_overlap(scribe_g_subgraph_n, _3dssg_g_subgraph_n, args.cos_sim_thr)
                #     # overlap_iter += 1 # just take a random one if no good overlap found after 10000 overlap_iter
                _3dssg_g_n = _3dssg_graphs[np.random.choice([k.split('_')[0] for k in current_keys if k.split('_')[0] != scribe_id.split('_')[0]])]
                scribe_g_subgraph_n, _3dssg_g_subgraph_n = get_matching_subgraph(scribe_g, _3dssg_g_n)
                if _3dssg_g_subgraph_n is None: _3dssg_g_subgraph_n = _3dssg_g_n

                scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g) # TODO: 3) check what the graph neural network is doing
                # overlap = calculate_overlap(scribe_g_subgraph, _3dssg_g_subgraph, args.cos_sim_thr)
                # if overlap < args.overlap_thr: print(f'Warning: positive pair overlap is less than threshold: {overlap}')

                # x = torch.tensor([scribe_g.nodes[i].features for i in scribe_g.nodes]).to('cuda') # TODO: Why is x not the same as x_node_ft?
                # p = torch.tensor([_3dssg_g.nodes[i].features for i in _3dssg_g.nodes]).to('cuda')

                x_node_ft, x_edge_idx, x_edge_ft = scribe_g.to_pyg() # scribe_g.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg() # _3dssg_g.to_pyg()
                n_node_ft, n_edge_idx, n_edge_ft = _3dssg_g_subgraph_n.to_pyg()
                if len(x_edge_idx[0]) <= 2 or len(p_edge_idx[0]) <= 2 or len(n_edge_idx[0]) <= 2: 
                    skipped += 1
                    continue

                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                x_n, n_n, m_n = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(n_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(n_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(n_edge_ft), dtype=torch.float32).to('cuda'))
                
                iter += 1
                curr_batch += 1

                loss1 = 1 - F.cosine_similarity(x_p, p_p, dim=0) # [0, 2] 0 is good
                loss2 = 2 - (1 - F.cosine_similarity(x_n, n_n, dim=0)) # [0, 2] 2 is good
                loss3 = (1 - m_p) + m_n

                loss += loss1 + loss2 + loss3
                
                if (curr_batch % batch_size == 0):
                    optimizer.zero_grad()
                    loss = loss / batch_size
                    epoch_loss = loss
                    loss.backward()
                    optimizer.step()
                    wandb.log({"loss1": loss1.item(),
                                "loss2": loss2.item(),
                                "loss3": loss3.sum().item(),
                                "loss": loss.item(),
                                "match_prob_pos": m_p.item(),
                                "match_prob_neg": m_n.item()})
                    loss = 0
                    curr_batch = 0

            wandb.log({"loss_per_epoch": epoch_loss.item()})
            if epoch % 10 == 0:
                evaluate_model(model, scanscribe_graphs_test, _3dssg_graphs, 'test')
                print(f'x_p first 10: {x_p[:10]}')
                print(f'x_n first 10: {x_n[:10]}')
                print(f'p_p first 10: {p_p[:10]}')
                print(f'n_n first 10: {n_n[:10]}')
            print(f'Skipped {skipped} graphs because one of the subgraphs had too few edges')
        return model


def evaluate_model(model, scanscribe, _3dssg, mode='test'):
    model.eval()
    valid_top_k = args.valid_top_k
    valid = {k: [] for k in valid_top_k}
    # TODO: Implement with top k option (1, 2, 4, 8) out of a test size of 32
    _3dssg = {k.split('_')[0]: _3dssg[k.split('_')[0]] for k in scanscribe}
    with torch.no_grad():
        for scribe_id in scanscribe:
            match_prob = []
            true_match = []
            scribe_g = scanscribe[scribe_id]
            for _3dssg_id in _3dssg:
                _3dssg_g = _3dssg[_3dssg_id]
                scribe_g_subgraph, _3dssg_g_subgraph = get_matching_subgraph(scribe_g, _3dssg_g)
                if _3dssg_g_subgraph is None: _3dssg_g_subgraph = _3dssg_g
                x_node_ft, x_edge_idx, x_edge_ft = scribe_g.to_pyg()
                p_node_ft, p_edge_idx, p_edge_ft = _3dssg_g_subgraph.to_pyg()
                x_p, p_p, m_p = model(torch.tensor(np.array(x_node_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_node_ft), dtype=torch.float32).to('cuda'),
                                        torch.tensor(x_edge_idx, dtype=torch.int64).to('cuda'), torch.tensor(p_edge_idx, dtype=torch.int64).to('cuda'),
                                        torch.tensor(np.array(x_edge_ft), dtype=torch.float32).to('cuda'), torch.tensor(np.array(p_edge_ft), dtype=torch.float32).to('cuda'))
                match_prob.append(m_p.item())
                if (scribe_id.split('_')[0] == _3dssg_id): true_match.append(1)
                else: true_match.append(0)
            
            # sort w indices
            match_prob = np.array(match_prob)
            true_match = np.array(true_match)
            sorted_indices = np.argsort(match_prob)
            match_prob = match_prob[sorted_indices]
            true_match = true_match[sorted_indices]
            print(f'match_prob: {match_prob}')
            print(f'true_match: {true_match}')
            for k in valid_top_k:
                if (1 in true_match[-k:]): valid[k].append(1)
                else: valid[k].append(0)
            # if (true_match[-1] == 1): valid.append(1)
            # else: valid.append(0)
    accuracy = {k: np.mean(valid[k]) for k in valid_top_k}
    # accuracy = np.mean(valid)
    for k in accuracy: wandb.log({f'accuracy_{str(mode)}_top{k}': accuracy[k]})
    print(f'accuracies: {accuracy}')
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='online')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('--overlap_thr', type=float, default=0.8)
    parser.add_argument('--cos_sim_thr', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--training_set_size', type=int, default=16)
    parser.add_argument('--test_set_size', type=int, default=16)
    parser.add_argument('--graph_size_min', type=int, default=6, help='minimum number of nodes in a graph')
    parser.add_argument('--contrastive_loss', type=bool, default=False)
    parser.add_argument('--valid_top_k', nargs='+', type=int, default=[1, 2, 4, 8])
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
            
    ######### Train / Test Split #########
    args.graph_size_min = 6

    # filter out graphs that are too small
    scanscribe_graphs = {k: scanscribe_graphs[k] for k in scanscribe_graphs if len(scanscribe_graphs[k].nodes) >= args.graph_size_min}
    print(f'num of graphs bigger than {args.graph_size_min}: {len(scanscribe_graphs)}')
    # sample in scanscribe
    scanscribe_graphs_train = {k: scanscribe_graphs[k] for k in random.sample(list(scanscribe_graphs), args.training_set_size)}
    scanscribe_graphs_test = {k: scanscribe_graphs[k] for k in random.sample([s for s in scanscribe_graphs if s not in list(scanscribe_graphs_train.keys())], args.test_set_size)}

    model = train_graph2graph(_3dssg_graphs, scanscribe_graphs_train)
    # val_3dssg = None
    # val_scribe = None
    # evaluate_model(model, val_3dssg, val_scribe)