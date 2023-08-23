import os
import sys

# go through 3RScan_descriptions folder
folder_names = os.listdir('../data/3DSSG/3RScan_descriptions')
folder_names.sort()

# for each folder
for folder_name in folder_names:
    # look for the same name in 3RScan
    if os.path.isdir('../data/3DSSG/3RScan/' + folder_name):
        # if found, copy 0.gif, 1.gif, 2.gif from 3RScan/folder_name/sequence_sampled to 3RScan_descriptions
        gif_names = ['0.gif', '1.gif', '2.gif']
        for gif_name in gif_names:
            os.system('cp ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + gif_name + ' ../data/3DSSG/3RScan_descriptions/' + folder_name + '/' + gif_name)