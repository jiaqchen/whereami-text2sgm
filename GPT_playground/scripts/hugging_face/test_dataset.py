# Go through /scanscribe.json and see if any of the scan_ids match with the scan_ids from relationships.json

import json
import os
import sys
import tqdm
import torch
import numpy as np

def check_3dssg_vs_scanscribe():
    # open relationships.json
    with open("../../../data/3DSSG/relationships.json", "r") as f:
        relationships = json.load(f)
    relationships = relationships['scans']

    # open scanscribe.json
    with open("./scanscribe.json", "r") as f:
        scanscribe = json.load(f)

    # print len
    print("Relationships len: ", len(relationships))
    print("Scanscribe len: ", len(scanscribe))

    # count unique scan_ids in scanscribe
    scan_ids = set()
    for s in scanscribe:
        scan_ids.add(s['scan_id'])
    print("Unique scan_ids in scanscribe: ", len(scan_ids))

    # for s in scanscribe:
    #     print(s['scan_id'])

    # check if any of the scan_ids match
    count = 0
    for r in relationships:
        for s in scanscribe:
            if r['scan'] == s['scan_id']:
                # print("Match found: ", r['scan'], s['scan_id'])
                count += 1
                break
    print("Total matches: ", count)

def clean(sentences):
    clean_sentences = []
    for s in sentences:
        # if substring contains 'sorry', remove it
        if 'sorry' in s or \
            'sorry,' in s or \
            'Without additional information' in s or \
            'you' in s or \
            'You' in s or \
            'You have already asked this question before' in s or \
            'You already asked me this question earlier.' in s or \
            'I apologize,' in s or \
            'I am sorry,' in s or \
            'I\'m sorry,' in s or \
            'additional context' in s or \
            'additional context,' in s or \
            'designated' in s or \
            'usually' in s or \
            s.startswith('It') or \
            '?' in s or \
            'impossible' in s or \
            'unknown' in s or \
            len(s) == 0:
            continue
        else:
            clean_sentences.append(s)
    if len(clean_sentences) == 0:
        return []
    if len(clean_sentences) == 1 and clean_sentences[0] == "":
        return []
    return clean_sentences

# main
if __name__ == "__main__":
    # open scanscribe.json
    with open("./scanscribe.json", "r") as f:
        scanscribe = json.load(f)

    # count unique scan_ids in scanscribe
    scanscribe_dict = {}
    scan_ids = set()
    for s in scanscribe:
        scan_ids.add(s['scan_id'])
        if s['scan_id'] not in scanscribe_dict:
            scanscribe_dict[s['scan_id']] = []
        scanscribe_dict[s['scan_id']].append(s['sentence'])

    # count number of sentences
    count = 0
    for scan in scanscribe_dict:
        count += len(scanscribe_dict[scan])
    print("Count of sentences before ", count)

    # Create new subsets of datasets
    new_scanscribe_dict = {}
    for scan in tqdm.tqdm(scanscribe_dict):
        sentences = scanscribe_dict[scan]
        sentences = clean(sentences)
        if len(sentences) == 0:
            continue
        if len(sentences) == 1 and sentences[0] == "":
            continue
        if len(sentences) <= 4:
            sentence = " ".join(sentences)
            sentences.append(sentence)
        else:
            loop = (len(sentences)-4)*5
            end = min(len(sentences) - 2, 6)
            len_1 = len(sentences) - 1
            for _ in range(loop):
                num_samples = np.random.randint(2, end)
                indices = np.random.choice(len_1, num_samples, replace=False)
                sentence = " ".join([sentences[i] for i in indices])
                sentences.append(sentence)
        new_scanscribe_dict[scan] = sentences
    
    # count number of sentences
    count = 0
    for scan in new_scanscribe_dict:
        count += len(new_scanscribe_dict[scan])
    print("Count of sentences ", count)

    # Save as json
    with open("scanscribe_1.json", "w") as f:
        json.dump(new_scanscribe_dict, f, indent=4)
