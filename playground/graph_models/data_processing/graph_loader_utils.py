import numpy as np
import matplotlib.pyplot as plt

# Python suppress warnings from spaCy
import warnings
warnings.filterwarnings("ignore", message=r"\[W095\]", category=UserWarning)

import spacy
import en_core_web_lg
# nlp = spacy.load("en_core_web_md")
nlp = spacy.load("en_core_web_lg")

from create_text_embeddings import create_embedding

def plot_relation(obj1, obj2, ax, distance):
    centroid1 = np.array(obj1['obb']['centroid'])
    centroid2 = np.array(obj2['obb']['centroid'])
    
    # Plot line between centroids, use distance as a thickness
    ax.plot([centroid1[0], centroid2[0]], [centroid1[1], centroid2[1]], [centroid1[2], centroid2[2]], c='b', linewidth=distance) 

def draw_obb(corners, ax):
    ax.plot([corners[0, 0], corners[1, 0]], [corners[0, 1], corners[1, 1]], [corners[0, 2], corners[1, 2]], c='g')
    ax.plot([corners[0, 0], corners[2, 0]], [corners[0, 1], corners[2, 1]], [corners[0, 2], corners[2, 2]], c='g')
    ax.plot([corners[0, 0], corners[3, 0]], [corners[0, 1], corners[3, 1]], [corners[0, 2], corners[3, 2]], c='g')
    ax.plot([corners[1, 0], corners[4, 0]], [corners[1, 1], corners[4, 1]], [corners[1, 2], corners[4, 2]], c='g')
    ax.plot([corners[1, 0], corners[5, 0]], [corners[1, 1], corners[5, 1]], [corners[1, 2], corners[5, 2]], c='g')
    ax.plot([corners[2, 0], corners[4, 0]], [corners[2, 1], corners[4, 1]], [corners[2, 2], corners[4, 2]], c='g')
    ax.plot([corners[2, 0], corners[6, 0]], [corners[2, 1], corners[6, 1]], [corners[2, 2], corners[6, 2]], c='g')
    ax.plot([corners[3, 0], corners[5, 0]], [corners[3, 1], corners[5, 1]], [corners[3, 2], corners[5, 2]], c='g')
    ax.plot([corners[3, 0], corners[6, 0]], [corners[3, 1], corners[6, 1]], [corners[3, 2], corners[6, 2]], c='g')
    ax.plot([corners[4, 0], corners[7, 0]], [corners[4, 1], corners[7, 1]], [corners[4, 2], corners[7, 2]], c='g')
    ax.plot([corners[5, 0], corners[7, 0]], [corners[5, 1], corners[7, 1]], [corners[5, 2], corners[7, 2]], c='g')
    ax.plot([corners[6, 0], corners[7, 0]], [corners[6, 1], corners[7, 1]], [corners[6, 2], corners[7, 2]], c='g')

def bounding_box(obj, ax=None, plot=False):
    mat44 = np.eye(4)
    mat44[:3, :3] = np.array(obj['obb']['normalizedAxes']).reshape(3, 3).transpose()
    mat44[:3, 3] = obj['obb']['centroid']

    # Get corners
    X, Y, Z = obj['obb']['axesLengths']
    corners = [
        [0, 0, 0, 1],
        [X, 0, 0, 1],
        [0, Y, 0, 1],
        [0, 0, Z, 1],
        [X, Y, 0, 1],
        [X, 0, Z, 1],
        [0, Y, Z, 1],
        [X, Y, Z, 1]
    ]
    # Offset the box by half its dimensions so centroid is at origin
    corners = np.array(corners) - np.array([X / 2, Y / 2, Z / 2, 0])

    if plot:
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Set axis to same scale
        ax.set_xlim3d(-100, 100)
        ax.set_ylim3d(-100, 100)
        ax.set_zlim3d(-100, 100)

        # Transformed corners
        corners = np.array([mat44 @ c for c in corners])
        ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='r', marker='o')
        draw_obb(corners, ax)
    else:
        corners = np.array([mat44 @ c for c in corners])

        min = np.min(corners, axis=0)
        max = np.max(corners, axis=0)
        return min[:3], max[:3]

def get_obj_distance(obj1, obj2, objs):
    obj1 = objs[obj1]
    obj2 = objs[obj2]

    # plot(na1, na2)
    A_min, A_max = bounding_box(obj1)
    B_min, B_max = bounding_box(obj2)

    # Calculate distance based on nearest distances between bounding boxes
    u = np.array([max(0, x) for x in A_min - B_max])
    v = np.array([max(0, x) for x in B_min - A_max])
    dist = np.sqrt(np.sum(u * u) + np.sum(v * v))
    return dist

def get_ada(desc, hash):
    if desc in hash:
        return hash[desc], hash
    else:
        hash[desc] = create_embedding(desc)
    return hash[desc], hash

def get_word2vec(desc, hash):
    if desc == "":
        return np.zeros(300)
    if desc in hash:
        return hash[desc], hash
    else:
        hash[desc] = nlp(desc)[0].vector
    return hash[desc], hash