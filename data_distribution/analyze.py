# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from helper import load_scene_graphs, load_text_graphs, calculate_overlap
from tqdm import tqdm

# Load the scene graph and text graph data
scene_graphs = load_scene_graphs('../playground/graph_model/list_of_graph_3dssg_dict.pt')
text_graphs = load_text_graphs('../playground/graph_model/list_of_graph_scanscribe_gpt.pt')
file_prefix = './features_only/'
scene_graphs = list(scene_graphs.values())

# Initialize a list to store the overlap values
overlap_values_overall_scene_to_text = []
overlap_values_true_scene_to_text = []
overlap_values_false_scene_to_text = []
overlap_values_overall_text_to_scene = []
overlap_values_true_text_to_scene = []
overlap_values_false_text_to_scene = []

# Iterate through each pair of scene graph and text graph
for scene_graph in tqdm(scene_graphs):
    for text_graph in text_graphs:
        # Calculate the overlap between the nodes of the scene graph and text graph
        overlap_scene_to_text = calculate_overlap(scene_graph, text_graph)
        overlap_text_to_scene = calculate_overlap(text_graph, scene_graph)

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

''' TODO:
- [ ] refactor code!
- [ ] analysis with attributes
- [ ] analysis with spatial relationships
- [ ] analysis with nlp word2vec of the words
- [ ] analysis with subgraphing
- [ ] sampling from the true and false pairs
- [ ] retraining the models
'''