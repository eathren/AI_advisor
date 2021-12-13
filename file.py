import csv
import json
import os
from os.path import exists as file_exists

def write(path:str, data:json):
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read(path:str) -> json: 
    with open(path, "r+") as f:
        return json.load(f)

def does_file_exist(file):  # Might be a more elegant way to do this one.
    return file_exists(file)

