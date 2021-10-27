import os
import requests
from dotenv import load_dotenv
import csv

load_dotenv()

AA_key = os.getenv('AA_key')

"""
How this is going to work:
Set up a dict of each stock ticker, perhaps in a json file.
For each file, analyze their daily open/endpoints.
Try and find promising stocks from these. Find ones that have.

Training for the models might need to come from historical data. Looks like the AA API can only do one ID at a time. 
AA Free has a 500/day limit. 25/mo is the price for a unlimited daily calls. 
Stocks do daily adjusted or weekly adjusted?

http://www.fmlabs.com/reference/default.htm?url=SimpleMA.htm
"""


class StockData:
    def __init__(self):
        self.key = os.getenv('AA_key')
        self.data = []
        self.history = []

    def fetch_data(self):
        # Currently using nasdaq_ids.csv
        with open('data/nasdaq_ids.csv') as csv_file:
            csv_data = csv.reader(csv_file)

            for row in csv_data:
                self.data.append(row)

    def fetch_live_data(self):
        for row in self.data[1:2]:
            params = {'symbol': str(row[0]), 'apikey': str(AA_key)}
            print(row[0], AA_key)
            response = requests.get(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
                params=params)

            if response.status_code == 200:
                id_data = response.json()
                self.history.append(id_data)


if __name__ == '__main__':
    data = StockData()
data.fetch_data()
data.fetch_live_data()
print(data.history)
