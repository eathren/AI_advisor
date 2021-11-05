import os
import requests
from dotenv import load_dotenv
import csv
import random
import pandas as pd
import json
import matplotlib.pyplot as plt

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

"""
How this is going to work:
Set up a dict of each stock ticker, perhaps in a json file.
For each file, analyze their daily open/endpoints.
Try and find promising stocks from these. Find ones that have.

ACTIONS THAT NEED TO BE TAKEN!
Write functions for MA, MACD, RSI, STOCH RSI, and more!
This data exists in a pandas dictionary for speed. If that doesn't do it, we can try Numpy


Training for the models might need to come from historical data. Looks like the AA API can only do one ID at a time. 
AA Free has a 500/day limit. 25/mo is the price for a unlimited daily calls. 
Stocks do daily adjusted or weekly adjusted?

http://www.fmlabs.com/reference/default.htm?url=SimpleMA.htm
"""


class StockData:
    def __init__(self, id=None):
        self.id = id
        self.history = []
        self.json = self.fetch_json()
        self.data = self.fetch_stock()

    def fetch_json(self):
        params = {'symbol': self.id, 'apikey': AA_KEY}
        response = requests.get(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
            params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"An error has occured with fetching the json data for {self.id}")

    def fetch_stock(self) -> dict:
        data = pd.json_normalize(self.json)
        print(data)
        return data

    def write_data(self):
        file_path = "data/stocks/data/" + self.id + ".json"
        with open(file_path, 'w', encoding='utf-8') as f:
            # other choice to dump: self.data.to_json()
            json.dump(self.json, f, ensure_ascii=False, indent=4)

    # def fetch_all_names(self):
    #     # Currently using nasdaq_ids.csv
    #     with open('data/nasdaq_ids.csv') as csv_file:
    #         csv_data = csv.reader(csv_file)
    #         for row in csv_data:
    #             self.data.append(row)

    # def fetch_live_data(self):
    #     for row in self.data[1:2]:
    #         params = {'symbol': str(row[0]), 'apikey': str(AA_KEY)}
    #         print(row[0], AA_KEY)
    #         response = requests.get(
    #             "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
    #             params=params)
    #
    #         if response.status_code == 200:
    #             id_data = response.json()
    #             self.history.append(id_data)

    def print_random(self):
        """
        prints a random stock
        """
        print(self.data[random.randint(2, len(self.data) - 1)])


if __name__ == '__main__':
    stock = StockData("AMZN")
    # data.fetch_all_names()
    # stock.fetch_live_data()
    # print(stock.history)
    # stock.fetch_random()
    print(stock.json)
    stock.write_data()
