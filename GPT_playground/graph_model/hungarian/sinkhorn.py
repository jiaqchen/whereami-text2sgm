import torch
import argparse
import os
import sys
import tqdm
import random

sys.path.insert(0, '/home/julia/Documents/h_coarse_loc/GPT_playground/graph_model')

from sg_dataloader import SceneGraph

random.seed(0)

config = {
    'match_threshold': 0.0001,
    'iters': 100,
    'feature_dim': 1536,
}

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def optimal_transport(list_of_graph_text, dict_of_3dssg):
    # shuffle list_of_graph_text
    random.shuffle(list_of_graph_text)
    for text_graph in tqdm.tqdm(list_of_graph_text):
        # Get scene id
        scene_id = text_graph.scene_id
        # Get 3DSSG graph
        graph_3dssg = dict_of_3dssg[scene_id]
        # Get node features
        text_node_features = text_graph.get_node_features()
        text_node_features = torch.tensor(text_node_features)
        graph_node_features = graph_3dssg.get_node_features()
        graph_node_features = torch.tensor(graph_node_features)

        # Calculate score with einsum
        scores = torch.einsum('ij,kj->ik', text_node_features, graph_node_features)
        scores = scores / (config['feature_dim'])**.5
        scores = scores.unsqueeze(0)

        # Calculate optimal transport
        alpha = torch.tensor(1.)
        scores = log_optimal_transport(scores, alpha, config['iters'])

        assert(scores.shape == (1, text_node_features.shape[0] + 1, graph_node_features.shape[0] + 1))

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        output = {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

        # iterate with enumerate
        print("\n\nMatches from text to graph----------------------------------------------------")
        for idx, val in enumerate(indices0[0]):
            if (val != -1):
                print(text_graph.get_nodes()[idx].label, " --> ", graph_3dssg.get_nodes()[val].label)
        print("\n\nMatches from graph to text----------------------------------------------------")
        for idx, val in enumerate(indices1[0]):
            if (val != -1):
                print(graph_3dssg.get_nodes()[idx].label, " --> ", text_graph.get_nodes()[val].label)

        # curtain text
        curtain = text_graph.get_nodes()[0]
        curtain2 = graph_3dssg.get_nodes()[3]
        assert(curtain.label == 'curtain' and curtain2.label == 'curtain')

        # norm of curtain - curtain2 features
        print("\n\nNorm of curtain and curtain2 features-----------------------------------------")
        print(torch.norm(text_node_features[0] - graph_node_features[3]))

        print("\n\nNorm of random-------------------------------------------------------------------------")
        print(torch.norm(text_node_features[0] - graph_node_features[0]))

    
        # print all nodes
        print("\n\nText nodes---------------------------------------------------------------------")
        for node in text_graph.get_nodes():
            print(node.label, end=", ")
        print("\n\nGraph nodes--------------------------------------------------------------------")
        for node in graph_3dssg.get_nodes():
            print(node.label, end=", ")
        print("\n\n")
        print(output)
        exit()

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_source', type=str, default='scanscribe')
    args = parser.parse_args()

    bin_score = torch.nn.Parameter(torch.tensor(1.))

    # Load graphs
    list_of_graph_text = None
    list_of_graph_3dssg = None
    # Load scanscribe as graphs with node features in word2vec embedding space
    if (args.text_source == 'scanscribe'):
        print("Using scanscribe --> text graph dataset")
        scene_ids = os.listdir('../../scripts/scanscribe_json_gpt')
        
        # TODO: Try adding attributes to the features and saving another graph checkpoint
        if os.path.exists('list_of_graph_scanscribe_gpt_noplacenode_ada_002.pt'):
            print("Using ScanScribe presaved text source")
            list_of_graph_text = torch.load('list_of_graph_scanscribe_gpt_noplacenode_ada_002.pt')
        else:
            # Go through folders
            list_of_graph_text = []
            for scene_id in tqdm.tqdm(scene_ids):
                # Get files in folder
                texts = os.listdir('../../scripts/scanscribe_json_gpt/' + scene_id)
                for text_i in texts:
                    # Load scene graph
                    try:
                        scene_graph_scanscribe_gpt = SceneGraph('human+GPT', scene_id, raw_json='../../scripts/scanscribe_json_gpt/' + scene_id + '/' + text_i) # ScanScribe3DSSG+GPT has the same JSON signature as human+GPT
                    except Exception as e:
                        # print(e)
                        # print(traceback.format_exc())
                        print("Error with loading ScanScribe3DSSG+GPT scene graph ", scene_id, "           text ", text_i)
                        continue

                    # Process graph such that there are no gaps in indices and all nodes index from 0
                    try:
                        scene_graph_scanscribe_gpt.to_pyg()
                    except Exception as e:
                        # print(e)
                        # print(traceback.format_exc())
                        print("Error with conversion to pyg graph for ScanScribe3DSSG+GPT scene graph ", scene_id, "           text ", text_i)
                        continue
                    list_of_graph_text.append(scene_graph_scanscribe_gpt)
                
            # Save list to file to access later
            torch.save(list_of_graph_text, 'list_of_graph_scanscribe_gpt_noplacenode_ada_002.pt')

    # We must have a list_of_graph_3dssg_noplacenode_ada_002
    if os.path.exists('list_of_graph_3dssg_noplacenode_ada_002.pt'):
        print("Using 3DSSG presaved scene graphs")
        list_of_graph_3dssg = torch.load('list_of_graph_3dssg_noplacenode_ada_002.pt') 
    else: # Load 3DSSG graphs as dict
        scene_ids_3dssg = os.listdir('../../../data/3DSSG/3RScan')
        list_of_graph_3dssg = {}
        for scene_id in tqdm.tqdm(scene_ids_3dssg):
            try:
                scene_graph_3dssg = SceneGraph('3DSSG', scene_id, euc_dist_thres=1.0)
            except Exception as e:
                print("Error with loading 3DSSG scene graph scene ", scene_id)
                continue
            try:
                scene_graph_3dssg.to_pyg()
            except:
                continue
            list_of_graph_3dssg[scene_id] = scene_graph_3dssg
        torch.save(list_of_graph_3dssg, 'list_of_graph_3dssg_noplacenode_ada_002.pt')

    # Match nodes using optimal transport
    optimal_transport(list_of_graph_text, list_of_graph_3dssg)