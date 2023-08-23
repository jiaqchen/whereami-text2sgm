# Go through 3RScan and sample different sequences
import os
import random
import cv2
import imageio
from PIL import Image, ImageDraw
import sys
import tqdm
import argparse

# main
if __name__ == '__main__':
    # go through 3RScan data folder
    folder_names = os.listdir('../data/3DSSG/3RScan')
    folder_names.sort()

    # sample random folders
    folder_names = random.sample(folder_names, 10)

    # for each folder
    for folder_name in ['0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca']:
    # for folder_name in tqdm.tqdm(folder_names):
        # open the /sequence_sampled/ folder
        gifs = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled')

        # only get files named x.gif
        gifs = [gif for gif in gifs if gif[-4:] == '.gif']
        gifs.sort()

        # don't open if it has 300ms in the name
        gifs = [gif for gif in gifs if '300ms' not in gif]
        print(gifs)

        # open each gif and display it, and wait for user input
        for gif in gifs:
            # open gif
            gif = imageio.mimread('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + gif)

            # display gif
            imageio.mimsave('./test.gif', gif, duration=0.3)
            os.system('open ./test.gif')

            # wait for user input
            input('Press enter to continue...')

            # close gif
            os.system('killall Preview')