import json
from os.path import exists as file_exists


def store_json(file, data):
    with open(file, 'w+') as f:
        json.dump(data, f, indent=4)


def load_json(file):
    with open(file, "r+") as f:
        return json.load(f)


def does_file_exist(file):  # Might be a more elegant way to do this one.
    return file_exists(file)

#  Make a class. If the file exists, fetch the data. Else, populate with a default instance and then work on it/save that to data.
