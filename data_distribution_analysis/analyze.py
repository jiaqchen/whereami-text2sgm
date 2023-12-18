import numpy as np
import matplotlib.pyplot as plt
from helper import load_scene_graphs, load_text_graphs, calculate_overlap, get_matching_subgraph
from tqdm import tqdm
import torch
import sys 

sys.path.insert(0, '/home/julia/Documents/h_coarse_loc/playground')
import graph_models.data_processing.sg_dataloader as sg_dataloader

# Load the scene graph and text graph data
path_to_playground = '/home/julia/Documents/h_coarse_loc/playground'
path_to_data_distribution_analysis = '/home/julia/Documents/h_coarse_loc/data_distribution_analysis'
# scene_graphs = load_scene_graphs(path_to_playground + '/graph_models/data_checkpoints/list_of_graph_3dssg_dict.pt')
# text_graphs = load_text_graphs(path_to_playground + '/graph_models/data_checkpoints/list_of_graph_scanscribe_gpt.pt')
scene_graphs = load_scene_graphs(path_to_data_distribution_analysis + '/temp_data_processed/list_of_scene_subgraphs_no_attrib.pt')
text_graphs = load_text_graphs(path_to_data_distribution_analysis + '/temp_data_processed/list_of_text_subgraphs_no_attrib.pt')
file_prefix = './distributions/subgraphing_no_attrib/'
# scene_graphs = list(scene_graphs.values())

# Check that scene graphs have nodes
num_good = 0
for scene_graph in scene_graphs:
    if len(scene_graph.nodes) > 0:
        num_good += 1
num_good_text = 0
for text_graph in text_graphs:
    if len(text_graph.nodes) > 0:
        num_good_text += 1

# Initialize a list to store the overlap values
overlap_values_overall_scene_to_text = []
overlap_values_true_scene_to_text = []
overlap_values_false_scene_to_text = []
overlap_values_overall_text_to_scene = []
overlap_values_true_text_to_scene = []
overlap_values_false_text_to_scene = []

# Saving data again but with subgraphs instead
subgraphs = {}
scene_graphs_subgraphed = {}
text_graphs_subgraphed = {}

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

print("Len of scene graphs: " + str(len(scene_graphs)))
print("Len of text graphs: " + str(len(text_graphs)))

list_of_scene_subgraphs = []
list_of_text_subgraphs = []
overlap_failed = 0
# Iterate through each pair of scene graph and text graph, zipped
for scene_graph, text_graph in tqdm(zip(scene_graphs, text_graphs)):
# for scene_graph in tqdm(scene_graphs):
    # for text_graph in text_graphs:
#################################################################### subgraphing and then saving subgraphs
#         # Do subgraphing
#         scene_graph_subgraphed, text_graph_subgraphed = get_matching_subgraph(scene_graph, text_graph)
#         list_of_scene_subgraphs.append(scene_graph_subgraphed)
#         list_of_text_subgraphs.append(text_graph_subgraphed)

# # Save as lists of subgraphs
# torch.save(list_of_scene_subgraphs, './temp_data_processed/list_of_scene_subgraphs_no_attrib.pt')
# torch.save(list_of_text_subgraphs, './temp_data_processed/list_of_text_subgraphs_no_attrib.pt')

# exit()

####################################################################

    # Calculate the overlap between the nodes of the scene graph and text graph
    try:
        overlap_scene_to_text = calculate_overlap(scene_graph, text_graph)
        overlap_text_to_scene = calculate_overlap(text_graph, scene_graph)
    except:
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
        overlap_values_false_scene_to_text.append(overlap_scene_to_text)
        overlap_values_false_text_to_scene.append(overlap_text_to_scene)

# Convert the overlap values to a numpy array for further analysis
overlap_values_overall_scene_to_text = np.array(overlap_values_overall_scene_to_text)
overlap_values_true_scene_to_text = np.array(overlap_values_true_scene_to_text)
overlap_values_false_scene_to_text = np.array(overlap_values_false_scene_to_text)
overlap_values_overall_text_to_scene = np.array(overlap_values_overall_text_to_scene)
overlap_values_true_text_to_scene = np.array(overlap_values_true_text_to_scene)
overlap_values_false_text_to_scene = np.array(overlap_values_false_text_to_scene)

# Draw histograms of the overlap values
plt.figure()
plt.hist(overlap_values_overall_scene_to_text, bins=10)
plt.title('Overall overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'overall_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_true_scene_to_text, bins=10)
plt.title('True overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'true_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_false_scene_to_text, bins=10)
plt.title('False overlap from scene to text')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'false_overlap_scene_to_text_features_only.png')

plt.figure()
plt.hist(overlap_values_overall_text_to_scene, bins=10)
plt.title('Overall overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'overall_overlap_text_to_scene_features_only.png')

plt.figure()
plt.hist(overlap_values_true_text_to_scene, bins=10)
plt.title('True overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'true_overlap_text_to_scene_features_only.png')

plt.figure()
plt.hist(overlap_values_false_text_to_scene, bins=10)
plt.title('False overlap from text to scene')
plt.xlabel('Overlap')
plt.ylabel('Frequency')
plt.savefig(file_prefix + 'false_overlap_text_to_scene_features_only.png')

# Make text file
with open(file_prefix + 'overall.txt', 'w') as f:
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
- [ ] recreate the model checkpoints to make sure the labeling are all correct, currently missing a 
      multi combo for text dataset, for no_attrib. (i think), the one with combo right now is w_attrib
- [ ] analysis with spatial relationships
- [X] analysis with nlp word2vec of the words (analysis without attributes)
- [X] analysis with subgraphing
- [ ] rewrite the visualizer
- [ ] sampling from the true and false pairs
- [ ] retraining the models
- [ ] using attributes improves "false-pair" graph matching distirbution, but decreases "true-pair" graph matching distribution
      are attributes actually helping?
'''