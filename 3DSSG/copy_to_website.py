# Go through the sequence_sampled and copy gifs to website folder
import os
import tqdm

# open the sequence_sampled folder
scans = os.listdir('../data/3DSSG/3RScan/')
scans.sort()

print(scans[0])

for scan in tqdm.tqdm(scans):
    # if not exists, mkdir ../data_website/data/scan
    if not os.path.exists('../data_website/data/' + scan):
        os.system('mkdir ../data_website/data/' + scan)

    # open the sequence_sampled folder
    sequence_sampled = os.listdir('../data/3DSSG/3RScan/' + scan + '/sequence_sampled/')
    sequence_sampled.sort()

    # only take files ending with 300ms.gif
    sequence_sampled = [gif for gif in sequence_sampled if gif.endswith('300ms.gif')]

    # copy the gif to website folder
    for gif in sequence_sampled:
        os.system('cp ../data/3DSSG/3RScan/' + scan + '/sequence_sampled/' + gif + ' ../data_website/data/' + scan + '/' + gif)

