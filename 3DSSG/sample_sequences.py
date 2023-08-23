# Go through the /data/3DSSG/3RScan/scene_id/sequence/ folder and sample x number of the *.color.jpg files
import os
import sys
import tqdm
import random
import argparse


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_sequences', type=int, default=3, help='number of subsequences of images to sample') # Some folders just have less than 10 images, so sequences might all just be the same
    parser.add_argument('--length_sequence', type=int, default=100, help='length of each subsequence of images')

    num_sequences = parser.parse_args().num_sequences
    length_sequence = parser.parse_args().length_sequence

    # read folder names in /data/3DSSG/3RScan
    folder_names = os.listdir('../data/3DSSG/3RScan')
    folder_names.sort()


    # tqdm through the folder names
    # for folder_name in tqdm.tqdm(folder_names):
    for folder_name in ['0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca']:
        # try to find sequence.zip folder
        try:
            # read image names in /data/3DSSG/3RScan/scene_id/sequence/
            image_names = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence')

            # only take files that end with .color.jpg
            image_names = [image_name for image_name in image_names if image_name.endswith('.color.jpg')]
            image_names.sort()

            len_image_names = len(image_names) # number of images to sample from
            for num_sequence in range(0, num_sequences):
                # sample a subsequence of length length_sequence
                if len_image_names >= length_sequence:
                    start_index = random.randint(0, len_image_names - length_sequence)
                    sampled_image_names = image_names[start_index:start_index + length_sequence]
                else:
                    sampled_image_names = image_names
                
                # mkdir recursive /data/3DSSG/3RScan/scene_id/sequence_sampled/num_sequence if doesn't exist
                os.system('mkdir -p ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + str(num_sequence))

                # copy sampled images into /data/3DSSG/3RScan/scene_id/sequence_sampled/num_sequence/
                for sampled_image_name in sampled_image_names:
                    os.system('cp ../data/3DSSG/3RScan/' + folder_name + '/sequence/' + sampled_image_name + ' ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + str(num_sequence) + '/')
            

        # if not found, print error
        except:
            print('Error: ' + folder_name)
            continue