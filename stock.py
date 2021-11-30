import os
import pandas as pd
import matplotlib.pyplot as plt
import handle_json
from fetch_stock_data import StockData

"""
A file where some of the stock data might be moved. Maybe. 
"""


class Stock:
    def __init__(self, id):
        self.id = id.upper()
        self.data = {}

    def read_data(self):
        file_path = "data/stocks/data/" + id + ".json"
        try:
            if handle_json.file_exists(file_path):
                with open(file_path, "r") as f:
                    self.data = json.load(f)
        except exception:
            raise FileNotFoundError
