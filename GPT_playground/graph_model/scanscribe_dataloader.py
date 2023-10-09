import json
import os
import sys

# main
if __name__ == "__main__":
    # open scanscribe.json
    with open("../scripts/hugging_face/scanscribe.json", "r") as f:
        scanscribe = json.load(f)

