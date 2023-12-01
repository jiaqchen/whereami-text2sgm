import torch
from helper import load_scene_graphs, load_text_graphs, calculate_overlap
from tqdm import tqdm

pairs = {}
true_pairs = {}
neg_pairs = {}

# # Load the scene graph and text graph data
# scene_graphs = load_scene_graphs('../playground/graph_model/list_of_graph_3dssg_dict_w_attrib.pt')
# text_graphs = load_text_graphs('../playground/graph_model/list_of_graph_scanscribe_gpt_w_attrib.pt')
# scene_graphs = list(scene_graphs.values())

# for scene_graph in tqdm(scene_graphs):
#     for text_graph in text_graphs:
#         overlap_scene_to_text = calculate_overlap(scene_graph, text_graph)
#         overlap_text_to_scene = calculate_overlap(text_graph, scene_graph)

#         if scene_graph.scene_id not in pairs:
#             pairs[scene_graph.scene_id] = []
#         pairs[scene_graph.scene_id].append((text_graph.scene_id, overlap_scene_to_text, overlap_text_to_scene, scene_graph, text_graph))

#         if scene_graph.scene_id == text_graph.scene_id:
#             if scene_graph.scene_id not in true_pairs:
#                 true_pairs[scene_graph.scene_id] = []
#             true_pairs[scene_graph.scene_id].append((text_graph.scene_id, overlap_scene_to_text, overlap_text_to_scene, scene_graph, text_graph))
#         else:
#             if scene_graph.scene_id not in neg_pairs:
#                 neg_pairs[scene_graph.scene_id] = []
#             neg_pairs[scene_graph.scene_id].append((text_graph.scene_id, overlap_scene_to_text, overlap_text_to_scene, scene_graph, text_graph))

# # Save the pairs
# torch.save(pairs, './sampled_data_pairs/pairs_features_only.pt')
# torch.save(true_pairs, './sampled_data_pairs/true_pairs_features_only.pt')
# torch.save(neg_pairs, './sampled_data_pairs/neg_pairs_features_only.pt')

# Load the pairs
pairs = torch.load('./sampled_data_pairs/pairs_features_only.pt')
true_pairs = torch.load('./sampled_data_pairs/true_pairs_features_only.pt')
neg_pairs = torch.load('./sampled_data_pairs/neg_pairs_features_only.pt')


