import json
from os.path import exists as file_exists

"""
This file handles some logic to check if a file exists, and to write/read json data in those files.
"""

def store_json(file, data):
    with open(f"{file}.json", 'w+') as f:
        json.dump(data, f, indent=4)


def load_json(file):
    with open(f"{file}.json", "r+") as f:
        return json.load(f)


def does_file_exist(file):  # Might be a more elegant way to do this one.
    return file_exists(file)

