import torch
import clip
from PIL import Image
import json
import re
import os
import random
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp

def take_avg_across_scenes(score, all_sentence_scenes):
    # for every sentence, take the average of the scores of the sentences from the same scene
    assert(len(score) == len(all_sentence_scenes))
    scene_score = {}
    seen_scenes = []
    for i, scene in enumerate(all_sentence_scenes):
        if scene not in seen_scenes:
            seen_scenes.append(scene)
            scene_indices = [j for j, s in enumerate(all_sentence_scenes) if s == scene]
            temp_score = [score[scene_ind] for scene_ind in scene_indices]
            scene_score[scene] = sum(temp_score) / len(temp_score)
    return scene_score


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("test.jpeg")).unsqueeze(0).to(device)
# # string = "a room with a bed and lights and the bed is white and there's a light on the ceiling. there is a huge plant in the corner, the pillows on the bed are blue, grey and white. the bedside table is an interesting wood texture. there are books on the bedside table. there is a tapestry hanging on the wall that looks to be made of yarn. there's also a bedside table on the other side of the bed in the corner with a lamp above it. there's also a window behind the plant. there is a picture hanging in front of the bed. it appears the bed is grey and has a grey blanket on it, but also a white comforter. outside there are umbrellas and a wooden fense, a stone fense, and a stone wall. the floor of the room is a light wooden color and texture"
# string = ['a room with a bed and lights and the bed is white and there\'s a light on the ceiling', 'there is a huge plant in the corner, the pillows on the bed are blue, grey and white.', 'the bedside table is an interesting wood texture.', 'there are books on the bedside table.', 'there is a tapestry hanging on the wall that looks to be made of yarn.', 'a room with a bed and a TV', 'a cat']
# # string = string.split('.')
# text = clip.tokenize(string).to(device)
# # text = text.type(torch.float32)
# # text = text.mean(dim=0)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# # average the probabilities up until the last 2
# probs_top = probs[0][:-2].mean(axis=0)
# print("Label probs:", probs_top)  # prints: [[0.9927937  0.00421068 0.00299572]]
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
# exit()

scanscribe_graphs_test_path = '/home/julia/Documents/h_coarse_loc/playground/graph_models/data_checkpoints/processed_data/testing/scanscribe_graphs_test_final_no_graph_min.pt'
scanscribe_graphs_test = torch.load(scanscribe_graphs_test_path)
scanscribe_test_scenes = list(scanscribe_graphs_test.keys())

scanscribe_cleaned = '/home/julia/Documents/h_coarse_loc/data/scanscribe/data/scanscribe_cleaned.json'
# for every scene in ScanScribe, for every sentence, split the sentence by commas and periods.
scanscribe_cleaned = json.load(open(scanscribe_cleaned, 'r'))

scene_sentences_tuples = []
print('Getting sentences for each scene')
for scene in tqdm(scanscribe_cleaned):
    if scene in scanscribe_test_scenes:
        for id, sentence in enumerate(scanscribe_cleaned[scene]):
            sentence = re.split(r'[.,]', sentence)
            sentence = [s.strip() for s in sentence]
            sentence = [s for s in sentence if len(s) > 0]
            scene_sentences_tuples.append((scene + '_' + str(id), sentence))

sample_count = 100
scene_images_tuples = []
print('Getting images for each scene scanscribe')

# for scene in tqdm(scanscribe_test_scenes):
#     # Get corresponding 3RScan images folder: '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan'
#     scene_images_folder = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/' + scene + '/sequence'
#     scene_images = os.listdir(scene_images_folder)
#     scene_images = [os.path.join(scene_images_folder, image) for image in scene_images if image.endswith('.jpg')]
#     # sample 100 images if they exist
#     if len(scene_images) > sample_count:
#         scene_images = random.sample(scene_images, sample_count)
#     scene_images_encoded = []
#     for img in scene_images:
#         scene_images_encoded.append(model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)).detach().cpu().numpy())
#         # torch free mem
#     torch.cuda.empty_cache()
#     # scene_images_encoded = [model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)) for img in scene_images]
#     scene_images_tuples.append((scene_images, scene_images_encoded))
# torch.save(scene_images_tuples, '/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/scene_images_tuples.pt')
scene_images_tuples = torch.load('/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/scene_images_tuples.pt')

all_sentences = [sentence for scene, sentences in scene_sentences_tuples for sentence in sentences]
all_sentence_scenes = [scene for scene, sentences in scene_sentences_tuples for sentence in sentences]
assert(len(all_sentences) == len(all_sentence_scenes))
print(len(scene_sentences_tuples))


############### HUMAN
human_data_path = '/home/julia/Documents/h_coarse_loc/website_data_labeling/server/data.json'
# load each line as part of a list, just read the file
human_data = open(human_data_path, 'r').readlines()
human_data = [json.loads(line) for line in human_data]
print(human_data[0])

scene_sentences_tuples_human = []
print('Getting sentences for each scene, human')
scenes = []
human_scenes = []
for idx, text_data in enumerate(human_data):
    text_data['scanId'] = text_data['scanId'].split('.')[0] + '_' + str(idx)
    scenes.append(text_data['scanId'])
    human_scenes.append(text_data['scanId'].split('/')[0])

    sentence = text_data['description']
    sentence = re.split(r'[.,]', sentence)
    sentence = [s.strip() for s in sentence]
    sentence = [s for s in sentence if len(s) > 0]
    scene_sentences_tuples_human.append((text_data['scanId'], sentence))
all_sentences_human = [sentence for scene, sentences in scene_sentences_tuples_human for sentence in sentences]
all_sentence_scenes_human = [scene for scene, sentences in scene_sentences_tuples_human for sentence in sentences]
assert(len(all_sentences_human) == len(all_sentence_scenes_human))
assert(len(scenes) == len(set(scenes)))


# print(f'human {all_sentences_human[:10]}')
# print(f'scancsribe {all_sentences[:10]}')
# exit()
 

# # encode all sentences human
# all_sentences_encoded_human = []
# for s in tqdm(all_sentences_human):
#     all_sentences_encoded_human.append(model.encode_text(clip.tokenize(s).to(device)).detach().cpu().numpy())
#     torch.cuda.empty_cache()
# torch.save(all_sentences_encoded_human, '/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/all_sentences_encoded_human.pt')
all_sentences_encoded_human = torch.load('/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/all_sentences_encoded_human.pt')

## HUMAN images encoding
sample_count = 100
# scene_images_tuples_human = []
# print('Getting images for each scene')
# for scene in tqdm(human_scenes):
#     # Get corresponding 3RScan images folder: '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan'
#     scene_images_folder = '/home/julia/Documents/h_coarse_loc/data/3DSSG/3RScan/' + scene + '/sequence'
#     scene_images = os.listdir(scene_images_folder)
#     scene_images = [os.path.join(scene_images_folder, image) for image in scene_images if image.endswith('.jpg')]
#     # sample 100 images if they exist
#     if len(scene_images) > sample_count:
#         scene_images = random.sample(scene_images, sample_count)
#     scene_images_encoded = []
#     for img in scene_images:
#         scene_images_encoded.append(model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)).detach().cpu().numpy())
#         # torch free mem
#     torch.cuda.empty_cache()
#     # scene_images_encoded = [model.encode_image(preprocess(Image.open(img)).unsqueeze(0).to(device)) for img in scene_images]
#     scene_images_tuples_human.append((scene_images, scene_images_encoded))
# torch.save(scene_images_tuples_human, '/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/scene_images_tuples_human.pt')
scene_images_tuples_human = torch.load('/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/scene_images_tuples_human.pt')


# encode all sentences ScanScribe
# all_sentences_encoded = []
# for s in tqdm(all_sentences):
#     all_sentences_encoded.append(model.encode_text(clip.tokenize(s).to(device)).detach().cpu().numpy())
#     torch.cuda.empty_cache()
# torch.save(all_sentences_encoded, '/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/all_sentences_encoded.pt')
all_sentences_encoded = torch.load('/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/all_sentences_encoded.pt')
assert(len(all_sentences_encoded) == len(all_sentences))




def cos_sim(a, b):
    a = np.reshape(a, (512,))
    b = np.reshape(b, (512,))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


dataset = "scanscribe"
if dataset == "human":
    all_sentences_encoded = all_sentences_encoded_human
    all_sentence_scenes = all_sentence_scenes_human
    scene_sentences_tuples = scene_sentences_tuples_human
    all_sentences = all_sentences_human # not needed
    folder_name = 'image_best_desc_human'
    max_scores_per_scene_folder_name = 'max_scores_per_scene_human'
elif dataset == "scanscribe":
    folder_name = 'image_best_desc'
    max_scores_per_scene_folder_name = 'max_scores_per_scene'
else:
    print("please enter dataset name")
    exit()


def f(tuple_pair):
    scene_ids_img, img_encoded = tuple_pair
    for idx, image in enumerate(img_encoded):
        scene_id_img = scene_ids_img[idx]
        with torch.no_grad():
            cos_sims = []
            for s in all_sentences_encoded: 
                cos_sims.append(cos_sim(image, s))
            # take the average across scenes
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            cos_sims_by_scene_desc = take_avg_across_scenes(cos_sims, all_sentence_scenes)
            assert(len(cos_sims_by_scene_desc) == len(scene_sentences_tuples))
            # print(len(cos_sims_by_scene_desc))
            # images_in_scene_scores.append(cos_sims_by_scene_desc.values())
        if not os.path.exists(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{folder_name}/{scene_id_img.split("/")[-3]}'):
            os.makedirs(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{folder_name}/{scene_id_img.split("/")[-3]}')
        torch.save(cos_sims_by_scene_desc, f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{folder_name}/{"/".join([scene_id_img.split("/")[i] for i in [-3, -1]])}.pt')



def get_one_img_scene_to_desc_scene(folder):
    prefix = f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{folder_name}/'
    files = os.listdir(prefix + folder)
    files = [os.path.join(prefix + folder, file) for file in files]
    scores = [torch.load(file) for file in files]
    # print(len(scores)) # number of images
    # print(len(scores[0])) # number of unique descriptions
    m = [] # matrix of all images in one scene to all unique descriptions of multiple scenes
    for score in scores: m.append(list(score.values()))
    m = np.array(m)
    # print(m.shape)
    # score for this scene with all the description is just the maximum score for each description
    max_scores = m.max(axis=0)
    assert(len(max_scores) == len(scores[0]))
    if not os.path.exists(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{max_scores_per_scene_folder_name}/{folder}'):
        os.makedirs(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{max_scores_per_scene_folder_name}/{folder}')
    torch.save(max_scores, f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{max_scores_per_scene_folder_name}/{folder}/max_scores.pt')

    # # sort with indices, in reverse order
    # max_scores_ind = np.argsort(max_scores)[::-1]
    # max_scores = max_scores[max_scores_ind]
    # # get the description ID for the top 5 scores
    # max_scores_desc = [list(scores[0].keys())[i] for i in max_scores_ind]
    # # get all indices of first score with same folder name
    # max_scores_desc_without_desc_id = [max_scores_desc[i].split('_')[0] for i in range(len(max_scores_desc))]
    # idxs = []
    # for i, desc in enumerate(max_scores_desc_without_desc_id):
    #     if desc == folder:
    #         idxs.append(i)

def get_top(index_in_scores, all_max_scores):
    scores = [all_max_scores[i] for i, _ in index_in_scores]
    scores = np.array(scores)
    # sort with indices, in reverse order
    max_scores_ind = np.argsort(scores)[::-1]
    max_scores = scores[max_scores_ind]
    # get the description ID for the top score
    return max_scores[0], index_in_scores[max_scores_ind[0]][1]

def get_all_scores(scene_names, text_desc_ids):

    if dataset == "human":
        scene_names = set([scene.split('/')[0] for scene in text_desc_ids])

    all_max_scores = []
    for scene in scene_names:
        # open max score
        max_score = torch.load(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{max_scores_per_scene_folder_name}/{scene}/max_scores.pt')
        all_max_scores.append(max_score)
    all_max_scores = np.array(all_max_scores)



    # # check for the first line
    # scene = scene_names[2]
    # desc_ids = text_desc_ids
    # scores = all_max_scores[2]
    # # arg sort
    # max_scores_ind = np.argsort(scores)[::-1]
    # max_scores = scores[max_scores_ind]
    # # get the description ID for the top 5 scores
    # max_scores_desc = [desc_ids[i] for i in max_scores_ind]
    # # get all indices of first score with same folder name
    # max_scores_desc_without_desc_id = [max_scores_desc[i].split('_')[0] for i in range(len(max_scores_desc))]
    # idxs = []
    # for i, desc in enumerate(max_scores_desc_without_desc_id):
    #     if desc == scene:
    #         idxs.append(i)
    # print(idxs)
    # exit()




    # Check 1 line of scores
    line = random.choice(all_max_scores)
    # print mean
    print(np.mean(line))
    # print variance
    print(np.var(line))
    # print max
    print(np.max(line))
    # print min
    print(np.min(line))

    # do this for all scores
    # print mean down the rows
    print(np.mean(all_max_scores, axis=0)[:5])
    # print variance down the rows
    print(np.var(all_max_scores, axis=0)[:5])
    # print max down the rows
    print(np.max(all_max_scores, axis=0)[:5])
    # print min down the rows
    print(np.min(all_max_scores, axis=0)[:5])

    img_scene_ids = scene_names
    desc_scene_ids = text_desc_ids

    assert(len(img_scene_ids) == 55 or len(img_scene_ids) == 142)
    assert(len(desc_scene_ids) == 1116 or len(desc_scene_ids) == 147)
    assert(all_max_scores.shape == (55, 1116) or all_max_scores.shape == (142, 147))
    desc_scene_ids_by_scene = {}
    for i, scene in enumerate(desc_scene_ids):
        if dataset == "scanscribe":
            text_id = scene.split('_')[1]
            scene_id = scene.split('_')[0]
        elif dataset == "human":
            scene_id = scene.split('/')[0]
        if scene_id not in desc_scene_ids_by_scene:
            desc_scene_ids_by_scene[scene_id] = [(i, scene)]
        else:
            desc_scene_ids_by_scene[scene_id].append((i, scene))
    assert(len(desc_scene_ids_by_scene) == 55 or len(desc_scene_ids_by_scene) == 142)

    eval_iters = 10000
    sample = 50
    mini_sample = len(img_scene_ids)
    top = [1, 5, 10, 20, 30, 40]

    in_top_w_var = {k: [] for k in top} # should be 1000 values of accuracies
    for _ in range(eval_iters):
        in_top = {k: [] for k in top}
        for img_i, img_scene_id in enumerate(random.sample(img_scene_ids, sample)): # 100

            removed = list(desc_scene_ids_by_scene.keys())
            removed.remove(img_scene_id)
            sample_desc_ids = random.sample(removed, mini_sample-1)
            assertion_sample_desc_ids = sample_desc_ids.copy()
            assertion_sample_desc_ids.append(img_scene_id)
            assert(len(set(assertion_sample_desc_ids)) == mini_sample)

            # top matching desc
            top_match_score, top_match_text_id = get_top(desc_scene_ids_by_scene[img_scene_id], all_max_scores[img_i])
            sampled_tuple = [random.sample(desc_scene_ids_by_scene[sample_desc_id], 1)[0] for sample_desc_id in sample_desc_ids]
            scores = [ all_max_scores[img_i, desc_idx] for desc_idx, _ in sampled_tuple ]
            scores.append(top_match_score)
            assert(len(scores) == mini_sample)
            scores = np.array(scores)
            # sort with indices, in reverse order
            max_scores_ind = np.argsort(scores)[::-1]
            max_scores = scores[max_scores_ind]
            # print(max_scores)
            
            for k in in_top: in_top[k].append(9 in max_scores_ind[:k]) # last index should be the top match score

        assert(len(list(in_top.values())[0]) == sample)
        for k in in_top_w_var: in_top_w_var[k].append(sum(in_top[k]) / len(in_top[k]))
    
    in_top_w_var = {k: (sum(in_top_w_var[k]) / len(in_top_w_var[k]), np.var(in_top_w_var[k])) for k in in_top_w_var}
    print(in_top_w_var)




# p = mp.Pool(processes=mp.cpu_count())
# p.map(f, scene_images_tuples)
# p.close()
# p.join()


# scene_names = os.listdir(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{folder_name}/')
# p = mp.Pool(processes=mp.cpu_count())
# p.map(get_one_img_scene_to_desc_scene, scene_names)
# p.close()
# p.join()

def check(scene_names):
    for s in scene_names:
        prefix = '/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/image_best_desc/'
        files = os.listdir(prefix + s)
        files = [os.path.join(prefix + s, file) for file in files]
        scores = [torch.load(file) for file in files]
        # check keys of all scores are the same
        keys = [list(score.keys()) for score in scores]
        assert(all(keys[0] == key for key in keys))
        assert(len(keys[0]) == 1116)

scene_names = os.listdir(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/{max_scores_per_scene_folder_name}')
text_desc_ids = list(torch.load(f'/home/julia/Documents/h_coarse_loc/baselines/CLIP-to-CLIP/image_best_desc/0ad2d38f-79e2-2212-98d2-9b5060e5e9b5/frame-000002.color.jpg.pt').keys())
get_all_scores(scene_names, text_desc_ids)
