"""Run the evaluation script on each of the prediction bundles."""
import concurrent.futures
import os
import json
import subprocess
import time

# UUID for evaluation script
EVAL_UUID = "0xbb7f73"
# UUID for test dataset bundle
TEST_UUID = "0x7ecf93" 
TAG = "{}_mrqa_evaluation"

prediction_bundles_uuids = [
    "0x63119a"
]

for dataset in ["amazon", "new_wiki", "nyt", "reddit"]:
    for prediction_uuid in prediction_bundles_uuids:
        command = [
            "cl",
            "run",
            "evaluate.py:{}".format(EVAL_UUID),
            "dev.json:{}/{}.jsonl".format(TEST_UUID, dataset),
            "predictions.json:{}".format(prediction_uuid),
            "python3 evaluate.py dev.json predictions.json",
            "--tags",
            TAG.format(dataset),
            "--request-docker-image",
            "xshaun/matplotlib"
        ]
        print(" ".join(command))
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        print(output)
