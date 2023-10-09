# Go through /scanscribe.json and see if any of the scan_ids match with the scan_ids from relationships.json

import json
import os
import sys

# main
if __name__ == "__main__":
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