# Python suppress warnings from spaCy
import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# Set up a class for doing the matching between 3RScan_descriptions and 3DSSG scene graph
import json
import pandas as pd
import argparse
import numpy as np
import spacy

# random seed
np.random.seed(0)

import tqdm

from collections import deque

nlp = spacy.load("en_core_web_md")
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting

# import Scene Graph
from _3dssg_sg_loader import SceneGraph
from _3dssg_description_loader import ScanDescriptions

# Localizer class for doing matching between Scene Graph and 3RScan description
class Localizer():
    def __init__(self, scene_graph, scan_descriptions, description_id_to_match, classifier, max_steps=2):
        self.scene_graph = scene_graph
        self.scan_descriptions = scan_descriptions
        self.max_steps = max_steps

        description_nouns = self.get_nouns(classifier) # Nouns in the descriptions, key is id
        sg_nouns = self.scene_graph.nouns # Nouns in the scene graph, key is id, Dictionary
        sg_nouns_sorted = self.scene_graph.nouns_sorted # Nouns in the scene graph sorted by count, List of tuples
        # Check nouns is not empty
        if len(sg_nouns) == 0 or len(description_nouns) == 0:
            print("No nouns found in scene graph or descriptions")
            return

        matches = self.match(description_nouns, sg_nouns, description_id_to_match)

        if description_id_to_match != None:
            self.draw_matches(matches[description_id_to_match])
        else:
            # TODO: Draw all the matches for every description_id
            pass

    def draw_matches(self, match):
        # Plot in 3D
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # For normalization
        max_graph_value = 0
        for m in match:
            if match[m]['graph_value'] > max_graph_value:
                max_graph_value = match[m]['graph_value']

        # Take the log of 300 with base 2
        base = 500 ** (float(1) / max_graph_value)

        # Dictionary of label to color
        label_color = {}

        # Look at objects_in_scan for the centroid
        for m in match:
            m_obj = match[m]
            if m_obj['label'] in ['floor', 'wall', 'ceiling']:
                continue

            # Find object in objects_in_scan with 'id' == m['id']
            centroid = None
            for obj in self.scene_graph.objects_in_scan:
                if str(obj['id']) == str(m):
                    centroid = obj['obb']['centroid']
                    break

            if centroid == None:
                print("No centroid found for object with id: ", m['id'])
                continue

            # If label is not in label_color, generate a new color for scatter point
            if m_obj['label'] not in label_color:
                label_color[m_obj['label']] = np.random.rand(3,)

            # Draw the centroid, with color using label_color
            ax.scatter(centroid[0], centroid[1], centroid[2], c=label_color[m_obj['label']], marker='o', s=base**m_obj['graph_value'])
 
            # Draw edges between object and its 'adj_to' objects
            for adj in m_obj['adj_to']:
                # Find the adj object in objects_in_scan
                adj_centroid = None
                for obj in self.scene_graph.objects_in_scan:
                    if str(obj['label']) in ['floor', 'wall', 'ceiling']:
                        continue
                    if str(obj['id']) == str(adj):
                        adj_centroid = obj['obb']['centroid']
                        break
                
                if adj_centroid == None:
                    print("No centroid found for object with id: ", adj)
                    continue
 
                # Draw a line between the two centroids
                ax.plot([centroid[0], adj_centroid[0]], [centroid[1], adj_centroid[1]], [centroid[2], adj_centroid[2]], c='b')
 
        # Draw a legend using label_color
        for label in label_color:
            ax.scatter([], [], c=label_color[label], label=label)
        ax.legend()
 
        # Show the plot with no axes
        ax.set_axis_off()
        plt.show()

    def noun_in_list_of_nouns(self, noun, nouns, threshold=0.5):
        # Get word2vec of noun
        noun_vec = nlp(noun)[0].vector

        # Find the noun in nouns with the highest similarity, spacy similarity
        max_sim = 0
        max_sim_noun = None
        for n in nouns:
            # oun_vec = nlp(n)[0].vector
            sim = nlp(noun).similarity(nlp(n))
            if sim > max_sim:
                max_sim = sim
                max_sim_noun = n

        return max_sim_noun, max_sim > threshold

    def sort_nouns_by_count(self, curr_nouns, sg_nouns):
        # Sort the nouns in curr_nouns by their count in sg_nouns
        curr_nouns_sorted = []
        seen = []
        for noun in curr_nouns:
            # if noun in sg_nouns and noun not in seen:
            #     curr_nouns_sorted.append((noun, sg_nouns[noun]))
            #     seen.append(noun)
            # elif noun + 's' in sg_nouns and noun not in seen:
            #     curr_nouns_sorted.append((noun + 's', sg_nouns[noun + 's']))
            #     seen.append(noun)
            # elif noun[:-1] in sg_nouns and noun[:-1] not in seen:
            #     curr_nouns_sorted.append((noun[:-1], sg_nouns[noun[:-1]]))
            #     seen.append(noun)

            # noun is adjacently connected to a noun in sg_nouns
            sg_noun, found_similar = self.noun_in_list_of_nouns(noun, sg_nouns, threshold=0.7)

            # Print similar
            if found_similar:
                print(noun + " is similar to " + sg_noun, "with similarity: ", nlp(noun).similarity(nlp(sg_noun)))

            if found_similar and noun not in seen:
                curr_nouns_sorted.append((sg_noun, sg_nouns[sg_noun]))
                seen.append(noun)

            # else just skip the noun, because it's not in the "scene graph" anyways
            # TODO: deal with skipped nouns, find similar in graph by word2vec

        # Sort the nouns by count
        curr_nouns_sorted = sorted(curr_nouns_sorted, key=lambda x: x[1])
        return curr_nouns_sorted

    def recurse_stack(self, stack, graph_adj_list, nouns_in_desc, current_steps, max_steps=3):
        # If stack is empty, return
        if len(stack) == 0:
            return graph_adj_list

        # If reached max_steps, return
        if len(current_steps) >= max_steps:
            return graph_adj_list

        # Pop the first element in the stack
        curr_node = str(stack.pop())

        try:
            # print("Graph_adj_list: ", graph_adj_list)
            curr_node_noun = graph_adj_list[curr_node]['label']
            max_sim_noun, found_similar = self.noun_in_list_of_nouns(curr_node_noun, nouns_in_desc, threshold=0.85)

            # If found similar, increment graph_value by 1 / count of noun in scene graph
            if found_similar:
                graph_adj_list[curr_node]['graph_value'] += 1 + (float(1) / len(self.scene_graph.label_id_mapping[max_sim_noun]))
        except KeyError:
            # print("Node " + str(curr_node) + " not in graph, or node is secondary in graph (it is a second_object in relationship)")
            return graph_adj_list

        current_steps.append(curr_node)

        # Get the children of the current node
        children = graph_adj_list[curr_node]['adj_to']
       
        # print("Children: ", children)
        for c in children:
            # If the child is not in the current steps, add to stack
            if c not in current_steps and c not in stack:
                stack.append(str(c))
                self.recurse_stack(stack.copy(), graph_adj_list, nouns_in_desc, current_steps.copy(), self.max_steps)
 
        return graph_adj_list
 
    # Input: desc_nouns: Dictionary of description id and nouns
    #        sg_nouns: Dictionary of scene graph nouns and their counts
    #        sg_nouns_sorted: List of tuples of scene graph nouns sorted by count
    #        description_id_to_match: Description id to match
    # Output: List of tuples of description id and scene graph id
    def match(self, desc_nouns, sg_nouns, description_id_to_match=None):
        # Iterate through descriptions
        if description_id_to_match != None:
            description_ids_to_match = [description_id_to_match]
        else:
            # Access the scan_descriptions.descriptions and get the "scene_id" in each description
            description_ids_to_match = [des['description_id'] for des in self.scan_descriptions.descriptions]

        matches = {}
 
        print("sg_nouns: ", sg_nouns)
        # Iterate through the description ids
        for description_id in description_ids_to_match:
            # Deep copy the scene graph adj list
            graph_adj_list = self.scene_graph.graph_adj_list.copy()
            # Get the nouns in the description
            curr_nouns = desc_nouns[description_id]
            # Sort the nouns in desc_nouns by their count in sg_nouns
            curr_nouns_sorted = self.sort_nouns_by_count(curr_nouns, sg_nouns)
            print("Nouns in description sorted: ", curr_nouns_sorted)
 
            nouns_in_desc_list = [x for (x, num) in curr_nouns_sorted]
 
            for n, n_count in curr_nouns_sorted:
                # print("Currently expanding noun: ", n)
                ids = list(self.scene_graph.label_id_mapping[n])
                # print("ids: ", ids)
 
                for l_id in ids:
                    stack = deque([l_id])
                    # print("Stack from label: ", stack)
               
                    graph_adj_list = self.recurse_stack(stack.copy(), graph_adj_list, nouns_in_desc_list, [], max_steps=2)
 
            # Add graph_adj_list to a dict with description_id as key and graph_adj_list as value
            matches[description_id] = graph_adj_list
 
        return matches
   
    def get_nouns(self, classifier):
        # TODO: Get the nouns from the descriptions
        # Dictionary of description id and nouns
        # Example: {'1': ['chair', 'table'], '2': ['chair', 'table']}
 
        nouns_dict = {}
        # Iterate through scan_descriptions
        for description in self.scan_descriptions.descriptions:
            # Get the description id
            description_id = description['description_id']
           
            # Import nlp library for noun extraction
            # nlp = spacy.load("en_core_web_md")
            doc = nlp(description['description'])
            nouns = []
            for token in doc:
                if token.pos_ == 'NOUN':
                    # If classifier predicts concrete noun
                    if classifier.predict([token.vector])[0] == 0:
                        nouns.append(token.text)
 
            # Add to dictionary
            nouns_dict[description_id] = nouns
 
        # Nouns key is description id
        return nouns_dict
   
 
def train():
    classes = ['concrete', 'abstract']
    # todo: add more examples
    train_set = [
        ['chair', 'couch', 'table', 'door', 'monitor', 'sink', 'cabinet', 'stuff', 'window', 'bed', 'lamp', 'shelf', 'toilet', 'mirror', 'picture', 'box', 'whiteboard', 'counter', 'desk', 'curtain', 'clothes', 'towel', 'fridge', 'shower', 'stool', 'fireplace', 'book', 'pillow', 'tv', 'plant', 'bottle', 'cup'],
        ['left', 'front', 'right', 'back', 'top', 'bottom', 'side', 'corner', 'middle', 'center', 'end', 'edge'],
    ]
    X = np.stack([list(nlp(w))[0].vector for part in train_set for w in part])
    y = [label for label, part in enumerate(train_set) for _ in part]
    classifier = LogisticRegression(C=0.1, class_weight='balanced').fit(X, y)
 
    for token in nlp("There is a hexagonal table in front of a whiteboard and a shelf. To the right of the table there is a large window. The table has red, blue, and yellow chairs around it."):
        if token.pos_ == 'NOUN':
            print(token, classes[classifier.predict([token.vector])[0]])
    return classifier
 
 
# Main function to load a Scene Graph, and a 3RScan description, and a new 3DSSG Localizer
if __name__ == "__main__":
    # Set up argparse to take in a scene_id
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, default='', help='scene_id')
    parser.add_argument('--description_id_to_match', type=str, default=None, help='description_id_to_match')
    parser.add_argument('--max_steps', type=int, default=2, help='max_steps')
    parser.add_argument('--euc_dist', type=float, default=1.5, help='euc_dist')
    args = parser.parse_args()
    scene_id = args.scene_id
    description_id_to_match = args.description_id_to_match
    max_steps = args.max_steps
    euc_dist = args.euc_dist
 
    if scene_id == '':
        print("Please enter a --scene_id")
        exit()
 
    # Train small nlp model
    classifier = train()
 
    # New Scene Graph
    scene_graph = SceneGraph(scene_id, euc_dist)
 
    # New 3RScan Description Loader
    scene_descriptions = ScanDescriptions(scene_id)

    # New Localizer
    localizer = Localizer(scene_graph, scene_descriptions, description_id_to_match, classifier, max_steps)