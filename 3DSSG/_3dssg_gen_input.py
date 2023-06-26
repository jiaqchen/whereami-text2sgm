# Go through the data/3DSSG/3RScan folder and count number of subfolders
import os
import json
import random
random.seed(0)

# Path for all objects
scenes_path = os.path.join(os.path.dirname(__file__), '../data/3DSSG/3RScan')

# Get all scenes
scenes = os.listdir(scenes_path)

# Get scenes from relationships.json
with open(os.path.join(os.path.dirname(__file__), '../data/3DSSG/relationships.json'), 'r') as f:
    relationships = json.load(f)
    relationships = relationships['scans']
    scenes_relationships = [scene['scan'] for scene in relationships]

count_good_scenes = 0
list_good_scenes = []
for s in scenes:
    if s in scenes_relationships:
        count_good_scenes += 1
        list_good_scenes.append(s)
print("Number of scenes in directory:", len(scenes))
print("Number of scenes in relationships.json:", len(scenes_relationships))
print(count_good_scenes)

def sample_and_write(num_samples, times=1000):
    # Randomly sample num_samples scenes times times
    random_scenes_all = []
    for i in range(times):
        # Sample without replacement
        random_scenes = random.sample(list_good_scenes, num_samples)
        random_scenes_all.append(random_scenes)

    with open(os.path.join(os.path.dirname(__file__), '../data/3DSSG/sampled_' + str(num_samples) + '_' + str(times) + '_scenes.txt'), 'w') as f:
        for random_scenes in random_scenes_all:
            random_scenes_str = str(random_scenes).strip('[]')
            f.write(random_scenes_str)
            f.write('\n')

# Randomly sample 4, 49, 99, 299 scenes and write their scene_id to a file
sample_and_write(4)
sample_and_write(49)
sample_and_write(99)
sample_and_write(299)

