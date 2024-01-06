import numpy as np
import matplotlib.pyplot as plt
from helper import load_scene_graphs, load_text_graphs, calculate_overlap, get_matching_subgraph
from tqdm import tqdm
import torch
import sys 

sys.path.insert(0, '/home/julia/Documents/h_coarse_loc/playground')
import graph_models.data_processing.scene_graph as scene_graph
from scene_graph import SceneGraph

######## 3DSSG ######### 1335 3DSSG graphs
_3dssg_graphs = {}
_3dssg_scenes = torch.load('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt')
for sceneid in tqdm(_3dssg_scenes):
    _3dssg_graphs[sceneid] = SceneGraph(sceneid, 
                                        graph_type='3dssg', 
                                        graph=_3dssg_scenes[sceneid], 
                                        max_dist=1.0, embedding_type='ada')

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
                                                                       embedding_type='ada')

save_files_dir = './distributions_post_refactor/subgraph_w_attrib_30-overlap/'

# Check that scene graphs have nodes
num_good = 0
for scene_graph_id in _3dssg_graphs:
    scene_graph = _3dssg_graphs[scene_graph_id]
    if len(scene_graph.nodes) > 0: num_good += 1
num_good_text = 0
for text_graph_id in scanscribe_graphs:
    text_graph = scanscribe_graphs[text_graph_id]
    if len(text_graph.nodes) > 0: num_good_text += 1
print(f'length of scene graphs: {len(_3dssg_graphs)} and num good: {num_good}')
print(f'length of text graphs: {len(scanscribe_graphs)} and num good: {num_good_text}')

# Initialize a list to store the overlap values
overlap_values_overall_scene_to_text = []
overlap_values_true_scene_to_text = []
overlap_values_false_scene_to_text = []
overlap_values_overall_text_to_scene = []
overlap_values_true_text_to_scene = []
overlap_values_false_text_to_scene = []

overlap_values_false_s2t_u90 = []
overlap_values_false_t2s_u90 = []

print("Len of scene graphs: " + str(len(_3dssg_graphs)))
print("Len of text graphs: " + str(len(scanscribe_graphs)))
print(f'len of zipped graphs: {len(list(zip(_3dssg_graphs, scanscribe_graphs)))}')

# save_overlap = {}
# for key in _3dssg_graphs: save_overlap[key] = {}
save_overlap = torch.load(save_files_dir + 'overlap_values.pt')
overlap_failed = 0
# Iterate through each pair of scene graph and text graph, zipped
# for scene_graph, text_graph in tqdm(zip(_3dssg_graphs, scanscribe_graphs)):
for sgid in tqdm(_3dssg_graphs):
    for tgid in tqdm(scanscribe_graphs):
        scene_graph = _3dssg_graphs[sgid]
        text_graph = scanscribe_graphs[tgid]
#################################################################### subgraphing and then saving subgraphs
        # scene_graph_subgraphed, text_graph_subgraphed = get_matching_subgraph(scene_graph, text_graph)
####################################################################
        # Calculate the overlap between the nodes of the scene graph and text graph
        try:
            # overlap_scene_to_text = calculate_overlap(scene_graph_subgraphed, text_graph_subgraphed)
            # overlap_text_to_scene = calculate_overlap(text_graph_subgraphed, scene_graph_subgraphed)
            overlap_scene_to_text = save_overlap[sgid][tgid]['overlap_scene_to_text']
            overlap_text_to_scene = save_overlap[sgid][tgid]['overlap_text_to_scene']
        except Exception as e:
            print(f'overlap failed for {sgid} and {tgid} because {e}')
            overlap_failed += 1
            continue

        # Store the overlap value
        overlap_values_overall_scene_to_text.append(overlap_scene_to_text)
        overlap_values_overall_text_to_scene.append(overlap_text_to_scene)

        # Check if the scene graph and text graph are a true match
        if scene_graph.scene_id == text_graph.scene_id:
            overlap_values_true_scene_to_text.append(overlap_scene_to_text)
            overlap_values_true_text_to_scene.append(overlap_text_to_scene)
        else:
            if overlap_scene_to_text < 0.9: overlap_values_false_s2t_u90.append(overlap_scene_to_text)
            if overlap_text_to_scene < 0.9: overlap_values_false_t2s_u90.append(overlap_text_to_scene)
            overlap_values_false_scene_to_text.append(overlap_scene_to_text)
            overlap_values_false_text_to_scene.append(overlap_text_to_scene)

        # Save the overlap values
        save_overlap[sgid][tgid] = {
            'overlap_scene_to_text': overlap_scene_to_text,
            'overlap_text_to_scene': overlap_text_to_scene
        }

# torch.save(save_overlap, save_files_dir + 'overlap_values.pt')

# Convert the overlap values to a numpy array for further analysis
overlap_values_overall_scene_to_text = np.array(overlap_values_overall_scene_to_text)
overlap_values_true_scene_to_text = np.array(overlap_values_true_scene_to_text)
overlap_values_false_scene_to_text = np.array(overlap_values_false_scene_to_text)
overlap_values_overall_text_to_scene = np.array(overlap_values_overall_text_to_scene)
overlap_values_true_text_to_scene = np.array(overlap_values_true_text_to_scene)
overlap_values_false_text_to_scene = np.array(overlap_values_false_text_to_scene)

overlap_values_false_s2t_u90 = np.array(overlap_values_false_s2t_u90)
overlap_values_false_t2s_u90 = np.array(overlap_values_false_t2s_u90)

# Draw histograms of the overlap values
plt.figure()
plt.hist(overlap_values_overall_scene_to_text, bins=10)
plt.title('Overall overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'overall_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_true_scene_to_text, bins=10)
plt.title('True overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'true_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_false_scene_to_text, bins=10)
plt.title('False overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'false_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_overall_text_to_scene, bins=10)
plt.title('Overall overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'overall_overlap_text_to_scene_features_only.png')

plt.figure()
plt.hist(overlap_values_true_text_to_scene, bins=10)
plt.title('True overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'true_overlap_text_to_scene_features_only.png')

plt.figure()
plt.hist(overlap_values_false_text_to_scene, bins=10)
plt.title('False overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'false_overlap_text_to_scene_features_only.png')

plt.figure()
plt.hist(overlap_values_false_s2t_u90, bins=10)
plt.title('False overlap from scene to text under 0.9')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'false_overlap_scene_to_text_features_only_u90.png')

plt.figure()
plt.hist(overlap_values_false_t2s_u90, bins=10)
plt.title('False overlap from text to scene under 0.9')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(save_files_dir + 'false_overlap_text_to_scene_features_only_u90.png')

# Make text file
with open(save_files_dir + 'overall.txt', 'w') as f:
    f.write('Number of overall scene to text: ' + str(len(overlap_values_overall_scene_to_text)) + '\n')
    f.write('Number of overall text to scene: ' + str(len(overlap_values_overall_text_to_scene)) + '\n')
    f.write('Number of false scene to text: ' + str(len(overlap_values_false_scene_to_text)) + '\n')
    f.write('Number of false text to scene: ' + str(len(overlap_values_false_text_to_scene)) + '\n')
    f.write('Number of true scene to text: ' + str(len(overlap_values_true_scene_to_text)) + '\n')
    f.write('Number of true text to scene: ' + str(len(overlap_values_true_text_to_scene)) + '\n')
    f.write('Number of overlap failed: ' + str(overlap_failed) + '\n')
    f.write("Num good scene graphs: " + str(num_good) + '\n')
    f.write("Num good text graphs: " + str(num_good_text) + '\n')

''' TODO:
- [X] refactor code!
- [X] analysis with attributes
- [X] recreate the model checkpoints to make sure the labeling are all correct, currently missing a 
      multi combo for text dataset, for no_attrib. (i think), the one with combo right now is w_attrib
- [ ] analysis with spatial relationships
- [X] analysis with nlp word2vec of the words (analysis without attributes)
- [X] analysis with subgraphing
- [X] rewrite the visualizer
- [ ] sampling from the true and false pairs
- [ ] retraining the models
- [ ] using attributes improves "false-pair" graph matching distirbution, but decreases "true-pair" graph matching distribution
      are attributes actually helping?
'''









'''
subgraph_pairs = {
    [
        {
            'scene_id': 'scene_id',
            'scene_graph_id': 'scene_graph_id',
            'text_graph_id': 'text_graph_id'
            'overlap_score_labels_only': 0.0,
            'overlap_score_w_attrib': 0.0,
            'overlap_score_labels_attrib_spatial': 0.0
        }
    ]
}
'''