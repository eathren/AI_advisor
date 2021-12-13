import csv
import json
import os
import random
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

from apis import alpaca, alpha_advantage
from file_handling import read, write, file_exists
from dotenv import load_dotenv

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

RISER_THRESHOLD = -5
FALLER_THRESHOLD = 6

CustomStrategy = ta.Strategy(
    name="Momo and Volatility",
    description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
    ta=[
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "bbands", "length": 20},
        {"kind": "rsi"},
        {"kind": "macd", "fast": 8, "slow": 21},
        {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
    ]
)


class StockData:
    """
    name: StockData
    Params: id: string, stock name. Ex: "AAPL" -> Apple
    Params: full: boolean, this indicates whether to do a full historical data fetch
    """

    def __init__(self, id=None, full=False):
        self.id = id.upper()
        if full:  # full data is used for neural net. Alpaca API only gets 1000 entries and runs much faster, so it is for score calculation.
            self.df = alpha_advantage.get_response(name=self.id)
        else:
            self.df = alpaca.get_response(name=self.id).astype(float)
        self.score = 0

    # Getter methods
    def get_id(self) -> str:
        return str(self.id)

    def get_score(self) -> int:
        return int(self.score)

    # Popultes df with indicators to use for analysis
    def populate_df_with_indicators(self):
        df = self.df
        df.ta.cores = 8  # How many cores to use.
        # applies the custom strategy to our dataframe.
        df.ta.strategy(CustomStrategy, append=True)

        # rsi = df.ta.rsi(close='close', length=14, append=True)
        # macd = df.ta.macd(close='close', fast=12, slow=26,
        #                   signal=9, append=True, signal_indicators=True)

        # cci = df.ta.cci(high='high', low='low', close='close', append=True)

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

        cci_val = df['CCI_14_0.015'].iloc[-1]
        rsi_val = rsi.iloc[-1]

        # df = df.ta.strategy(timed=True)  # This populates ALL INDICATORS
        # rsi oscillator check

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

        self.score = score
        return self.id, score, rsi_val, cci_val

    def print_df(self):
        print(self.df)

    def plot_df(self):
        ax = plt.gca()
        plt.plot(self.df)
        plt.show()

    @staticmethod
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
    # calc_all_risers_and_fallers()  # this populates the risers and fallers list
    stock = StockData("AAPL", full=False)
    stock.populate_df_with_indicators()
    stock.print_df()
    stock.plot_df()
