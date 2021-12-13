import csv
import json
import os
from os.path import exists


def write(path: str, data: json):
    with open(path, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read(path: str) -> dict:
    with open(path, "r+") as f:
        return json.load(f)


def file_exists(file):
    return exists(file)


def check_and_read(path: str) -> dict:
    """
    Name: check_and_read
    Params: path, str. This is the path to the file. 
    Returns: dict of file json data. 
    """
    try:
        if file_exists(path):
            return read(path)
    except IOError:
        print(f"The file at path: {path} does not exist.")
