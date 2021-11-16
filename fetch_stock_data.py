import os
import requests
from dotenv import load_dotenv
import csv
import random
import pandas as pd
import numpy as np
import json
import yfinance as yf
from time import sleep

import handle_json
import matplotlib.pyplot as plt

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

"""
How this is going to work:
Set up a dict of each stock ticker, perhaps in a json file.
Start a neural net with default configurations. 
Run the stock's data through the neural net day by day. 
For each file, analyze their daily open/endpoints.
Try and find promising stocks from these.

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
        self.id = id.upper()
        self.data = self.read_data()
        self.df = self.json_to_pd_df()
        # I'm not sure if I like modifying data like this.
        # put all the data into the dataframe
        
        self.populate_df_with_indicators()
        self.sma_7d = self.calc_sma(7)
        self.sma_14d = self.calc_sma(14)
        self.sma_21d = self.calc_sma(21)

    def read_data(self) -> dict:
        """
        This function will read the data from the stock/data folder if it exists.
        If it doesn't it will fetch new data.

        :return dictionary of stock price data:
        """
        file_path = f"data/stocks/data/{self.id}.json"
        if handle_json.file_exists(file_path):
            with open(file_path, "r") as f:
                temp = json.load(f)
            return temp
        else:
            return self.fetch_json()
        return None

    def fetch_json(self):
        params = {'symbol': self.id, 'apikey': AA_KEY}
        response = requests.get(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
            params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"An error has occured with fetching the json data for {self.id}")

    def json_to_pd_df(self):
        """
        Turns the json data into a pandas datafram
        :param self.json, json stock data from Alpha Advantage.
        :return df, a PD dataframe object:
        """
        return pd.DataFrame(self.data['Time Series (Daily)'])

    def fetch_stock(self) -> dict:
        data = pd.json_normalize(self.json)
        return data

    def write_data(self):
        file_path = "data/stocks/data/" + self.id + ".json"
        with open(file_path, 'w', encoding='utf-8') as f:
            # other choice to dump: self.data.to_json()
            json.dump(self.json, f, ensure_ascii=False, indent=4)

    def calc_stoch_rsi(self):
        pass

    def calc_rsi(self):
        pass

    def calc_ema(self):
        self.df[str(interval) + '_Day_EMA'] = \
            self.df['adjusted close'].ewm(com=interval, min_periods=0, adjust=False,
                                          ignore_na=False).mean()

    def calc_sma(self, interval):
        return self.df['adjusted close'].rolling(window=interval).mean()

    def calc_ema(self, interval):
        return self.df['adjusted close'].ewm(com=interval, min_periods=0, adjust=False, ignore_na=False).mean()

    def calc_macd(self):
        pass

    def calc_obv(self):
        pass

    def calc_adx(self):
        pass

    def calc_ad_line(self):
        pass

    def plot_data(self):
        df = pd.DataFrame(self.json['Time Series (Daily)'])
        print(df)
        df = df.T.iloc[::-1]
        print(df)
        x = pd.to_datetime(df.index)
        y = df["2. high"]
        plt.title(self.id)
        plt.xticks(rotation=90)
        plt.xlabel("date")
        plt.ylabel("dollars")
        plt.plot(x, y)
        plt.plot(x, df["6. volume"])
        plt.show()

    def plot_yfinance(self):
        ticker = yf.Ticker(self.id)
        history = ticker.history(start="2020-01-01", end="2020-02-01")
        history.head()
        print(ticker.info)
        data = yf.download(self.id)
        data.tail()
        data['Close'].plot(figsize=(10, 5))

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
        all_stocks = self.fetch_all_names()
        print(all_stocks[random.randint(2, len(all_stocks) - 1)])

    def fetch_fresh_data(self):
        all_stocks = fetch_all_names()
        print(all_stocks)
        for i, name in enumerate(all_stocks[1:]):
            sleep(1)  # this should be 1 second.
            stock = StockData(name)
            stock.write_data()


# End class methods here
def fetch_all_names():
    # Currently using nasdaq_ids.csv
    all_stocks = []
    with open('data/nasdaq_ids.csv') as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            all_stocks.append(row[0])

    return all_stocks


"""
N: 1-1000 stocks
Z: 1001 - 2000
K: 2000 - 3000
"""
if __name__ == '__main__':
    '''
    all_stocks = fetch_all_names()
    print(all_stocks)
    for i, name in enumerate(all_stocks[0:1000]):
        stock = StockData(name)
        stock.write_data()
    stock = StockData("TSLA")  # test value
        '''
        sleep(1) # this should be 1 second.
    # print(stock.json)
    stock.write_data()
    # stock.plot_data()
    # stock.print_random()
    # stock.plot_yfinance()
