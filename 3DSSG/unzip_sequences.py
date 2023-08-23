# Unzip some folders, and track progress
import os
import zipfile
import sys
import tqdm

# main 
if __name__ == '__main__':
    # read folder names in /data/3DSSG/3RScan
    folder_names = os.listdir('../data/3DSSG/3RScan')
    folder_names.sort()
    print(folder_names[0])

    # tqdm through the folder names
    for folder_name in tqdm.tqdm(folder_names):
        # try to find sequence.zip folder
        try:
            # unzip the folder inside called /sequence.zip, unzip into /sequence
            with zipfile.ZipFile('../data/3DSSG/3RScan/' + folder_name + '/sequence.zip', 'r') as zip_ref:
                zip_ref.extractall('../data/3DSSG/3RScan/' + folder_name + '/sequence')
        # if not found, print error
        except:
            print('Error: ' + folder_name)
            continue
