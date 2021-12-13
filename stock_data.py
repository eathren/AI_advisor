import csv
import json
import os
import random
from time import sleep

import handle_json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from apis import alpaca, alpha_advantage
from dotenv import load_dotenv

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

RISER_THRESHOLD = -5
FALLER_THRESHOLD = 6

class StockData:
    """
    name: StockData
    Params: id: string, stock name. Ex: "AAPL" -> Apple
    Params: full: boolean, this indicates whether to do a full historical data fetch
    """
    def __init__(self, id=None, full=False):
        self.id = id.upper()
        if full:  # full data is used for neural net. Alpaca API only gets 1000 entries.
            self.df = alpha_advantage.get_response(name=self.id)
        else:
            self.df = alpaca.get_response(name=self.id).astype(float)

    def get_id(self):
        return str(self.id)

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
        """
        name: fetch_json

        This function fetches all the stock data for a given stock ID.

        :return json stock data:
        """
        # This is a check on how much data to request from the API. compact is 30 days. Full is all.
        if self.full == True:
            size = 'full'
        else:
            size = 'compact'
        params = {'symbol': self.id, 'apikey': AA_KEY, 'outputsize': size}
        response = requests.get(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
            params=params)
        print(response.url)
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
        # Pandas is dumb when it comes to renaming rows. Make them columns to rename instead.
        df = pd.DataFrame(self.data['Time Series (Daily)']).transpose()
        # rename columns, since the api has a terrible naming convention.
        df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                                '5. adjusted close': 'close', '6. volume': 'volume',
                                '7. dividend amount': 'dividend amount', '8. split coefficient': 'split coefficient'})
        # change datatypes from Objects to floats
        df = df.astype(float)
        return df

    def fetch_stock(self) -> dict:
        data = pd.json_normalize(self.json)
        return data

    def write_data(self):
        file_path = "data/stocks/data/" + self.id + ".json"
        with open(file_path, 'w', encoding='utf-8') as f:
            data = self.data.to_json()
            json.dump(data, f, ensure_ascii=False, indent=4)

    def calc_if_riser_or_faller(self):
        """
        name: calc_if_riser_or_faller

        This function will iterate through all a stock pandas dataframe and return it's name, the score
        value, and rsi and cci indicators.

        :return self.id:string, score:int, rsi: float, cci:float
        """
        df = self.df
        score = 0  # this will be used to calculae a riser or faller.
        # score < 0 will be a possible faller.
        # score > 0 will be a possible riser.
        # Temporary rsi using pandas ta until ours works a little better.
        rsi = df.ta.rsi(close='close', length=14, append=True)
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True, signal_indicators=True)
        cci = df.ta.cci(high='high', low='low', close='close', append=True)
        cci_val = df['CCI_14_0.015'].iloc[-1]
        
        df.ta.cores = 8 # How many cores to use.
        df = df.ta.strategy(timed=True) # This populates ALL INDICATORS
        print(df.columns)
        # rsi oscillator check
        rsi_val = rsi.iloc[-1]

        if rsi_val > 90:
            score += 3
        elif 70 < rsi_val < 90:
            score += 2
        elif rsi_val > 70:
            score += 1
        elif 20 < rsi_val <= 30:
            score -= 1
        elif rsi_val < 20:
            score -= 3

        if cci_val > 200:
            score += 4
        elif 100 < cci_val < 200:
            score += 3
        elif cci_val > 100:
            score += 2
        elif cci_val < -100:
            score -= 2
        elif -200 < cci_val < -100:
            score -= 3
        elif cci_val < -200:
            score -= 4

        return self.id, score, rsi_val, cci_val

    def print_random(self):
        """
        prints a random stock
        """
        all_stocks = self.fetch_all_names()
        print(all_stocks[random.randint(2, len(all_stocks) - 1)])


def fetch_all_names():
    all_stocks = []
    with open('data/nasdaq_ids.csv') as csv_file:
        csv_data = csv.reader(csv_file)
        for row in csv_data:
            all_stocks.append(row[0])
    return all_stocks


def calc_all_risers_and_fallers():
    all_stocks = fetch_all_names()
    risers = {}
    fallers = {}
    for i, name in enumerate(all_stocks[1:]):
        try:
            stock = StockData(name, full=False)
            id, score, rsi, cci = stock.calc_if_riser_or_faller()
            print(f"id: {id}, score:{score}, rsi:{rsi}, cci:{cci}")
            if score >= FALLER_THRESHOLD:
                fallers[id] = {"score": score, "rsi": rsi, "cci": cci}
            if score <= RISER_THRESHOLD:
                risers[id] = {"score": score, "rsi": rsi, "cci": cci}
        except:
            print("Error 1: Something happened for stock: ", name)

    with open('data/stocks/risers/risers.json', 'w', encoding='utf-8') as f:
        json.dump(risers, f, ensure_ascii=False, indent=4)
    with open('data/stocks/fallers/fallers.json', 'w', encoding='utf-8') as f:
        json.dump(fallers, f, ensure_ascii=False, indent=4)
        
    return risers, fallers


if __name__ == '__main__':
    # fetch_fresh_data()  # this updates the data every day after closing
    # calc_all_risers_and_fallers()  # this populates the risers and fallers list
    stock = StockData("AAPL", full=True).df
    print(stock)
