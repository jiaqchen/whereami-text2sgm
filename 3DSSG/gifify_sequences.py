# Go through the /data/3DSSG/3RScan/scene_id/sequence_sampled/ folder and turn each subsequence into a gif
import os
import cv2
import imageio
from PIL import Image, ImageDraw
import sys
import tqdm
import argparse

images = []

width = 200
center = width // 2
color_1 = (0, 0, 0)
color_2 = (255, 255, 255)
max_radius = int(center * 1.5)
step = 8

for i in range(0, max_radius, step):
    im = Image.new('RGB', (width, width), color_1)
    draw = ImageDraw.Draw(im)
    draw.ellipse((center - i, center - i, center + i, center + i), fill=color_2)
    images.append(im)

for i in range(0, max_radius, step):
    im = Image.new('RGB', (width, width), color_2)
    draw = ImageDraw.Draw(im)
    draw.ellipse((center - i, center - i, center + i, center + i), fill=color_1)
    images.append(im)

images[0].save('./test.gif', save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

# main
if __name__ == '__main__':
    # go through the /data/3DSSG/3RScan/scene_id/sequence_sampled/ folder
    folder_names = os.listdir('../data/3DSSG/3RScan')
    folder_names.sort()

    # for folder_name in ['0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca']:
    # for folder_name in folder_names:
    # tqdm folder_names
    for folder_name in tqdm.tqdm(folder_names):
        # try to find sequence_sampled folder
        try:
            # read subsequence names in /data/3DSSG/3RScan/scene_id/sequence_sampled/
            subsequence_names = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled')
            subsequence_names.sort()

            # only take folders from sequence_names
            subsequence_names = [subsequence_name for subsequence_name in subsequence_names if os.path.isdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name)]

            # for each subsequence
            for subsequence_name in subsequence_names:

                # turn images into gif using PIL Image
                # Get images
                image_names = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name)
                image_names.sort()

                # turn images into Image
                images = []
                for image_name in image_names:
                    image = Image.open('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/' + image_name)
                    image = image.convert('RGB')
                    # rotate image
                    images.append(image)

                # save gif
                # if file exists, delete it
                if os.path.exists('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '_300ms.gif'):
                    os.remove('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '_300ms.gif')
                images[0].save('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '_300ms.gif', save_all=True, append_images=images[1:], duration=300, loop=0)
#############################################
                # turn subsequence into gif
                # print('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/*.jpg')
                # print('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '.gif')
#############################################
                # # get number of images in subsequence_name folder
                # images = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name)
                # images.sort()

                # # use opencv and imageio to turn images into gif
                # gifify_images = []
                # for ima in images:
                #     image = cv2.imread('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/' + ima)
                #     frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     gifify_images.append(frame_rgb)
#############################################
                # # write gif
                # imageio.mimsave('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '.gif', gifify_images, duration=5000) # in ms

                # use ffmpeg to turn images into gif
                # os.system('ffmpeg -y -f image2 -framerate 1 -pattern_type glob -i ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/frame-*.color.jpg ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '.gif')

                # os.system('convert -delay 30 -loop 0 ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled_renamed/' + subsequence_name + '/*.jpg ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled_renamed/' + subsequence_name + '.gif')
                # os.system('convert -delay 30 -loop 0 ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/*.color.jpg ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '.gif')
                # os.system('convert -delay 30 -loop 0 ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/*.color.jpg ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '.gif')

        except Exception as e:
            # print exception message
            print(e)

            print('Error: ' + folder_name)
            continue




############### For renaming images ###############
        #     # for each subsequence
        #     for subsequence_name in subsequence_names:
        #         # Get images
        #         image_names = os.listdir('../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name)
        #         image_names.sort()

        #         # mkdir
        #         os.system('mkdir -p ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled_renamed/' + subsequence_name)

        #         # rename images to just .jpg, not .color.jpg, put into jpg folder
        #         for image_name in image_names:
        #             os.system('mv ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled/' + subsequence_name + '/' + image_name + ' ../data/3DSSG/3RScan/' + folder_name + '/sequence_sampled_renamed/' + subsequence_name + '/' + image_name[:-10] + '.jpg')

                

        # except:
        #     print('Error: ' + folder_name)
        #     continue