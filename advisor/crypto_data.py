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
import fear_and_greed
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

from apis import alpaca, alpha_advantage
import util
from dotenv import load_dotenv

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

RISER_THRESHOLD = -5
FALLER_THRESHOLD = 6

"""
check if volume is rising

"""

class CryptoData:
    """
    name: CryptoData
    Params: id: string, crypto name. Ex: "AAPL" -> Apple
    Params: full: boolean, this indicates whether to do a full historical data fetch
    """

    def __init__(self, id=None, full=False, plot=False):
        self.id = id.upper()
        self.fear_and_greed = fear_and_greed.get().value
        if full:  # full data is used for neural net. Alpaca API only gets 1000 entries and runs much faster, so it is for score calculation.
            self.df = alpha_advantage.get_response(name=self.id)
        else:
            self.df = alpaca.get_response(name=self.id).astype(float)
        self.score = 0

        self.populate_df_with_indicators()
        self.df.astype(float)
        
        if plot:
            self.plot_df(save=True)

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


        # used to populate the CryptoData.df for fields used for calculations.
        CustomStrategy = ta.Strategy(
            name="Momo and Volatility",
            description="SMA 50,200, BBANDS, RSI, MACD and Volume SMA 20",
            ta=[
                # {"kind": "sma", "length": 20},
                {"kind": "sma", "length": 50},
                {"kind": "sma", "length": 200},
                # {"kind": "bbands", "length": 20},
                {"kind": "rsi"},
                {"kind": "obv"},
                {"kind": "macd", "fast": 8, "slow": 21},
                {"kind": "cci"},
                # {"kind": "sma", "close": "volume", "length": 20, "prefix": "VOLUME"},
            ]
        )
        df.ta.strategy(CustomStrategy, append=True)
        # print(df.columns.tolist())

    def calc_if_riser_or_faller(self):
        """
        name: calc_if_riser_or_faller

        This function will iterate through all a crypto pandas dataframe and return it's name, the score
        value, and rsi and cci indicators.

        :return self.id:string, score:int, rsi: float, cci:float
        """
        df = self.df
        score = 0  # this will be used to calculae a riser or faller.
        # score < 0 will be a possible faller.
        # score > 0 will be a possible riser.
        df.ta.cores = 8  # How many cores to use.

        # self.populate_df_with_indicators()
        # macd = df.ta.macd(close='close', fast=12, slow=26,
        #                   signal=9, append=True, signal_indicators=True)


        cci_val = df['CCI_14_0.015'].iloc[-1]
        rsi_val = df['RSI_14'].iloc[-1]

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

    def plot_df(self, save=False):
        df = self.df

        # using the variable axs for multiple Axes
       
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,  gridspec_kw={'height_ratios': [2, 1, 1, 1]})
            fig.suptitle(f"{self.id}")
            plt.xlabel("Date")
            # using tuple unpacking for multiple Axes
    
            # plt.plot(df['OBV'], label="OBV")
            ax1.plot(df['close'], label="CLOSE", linewidth=1.0)
            # ax1.plot(df['BBL_20_2.0'], linewidth=.3)
            # ax1.plot(df['BBP_20_2.0'], linewidth=.3)
            ax1.plot(df['SMA_50'], label="SMA_50", linewidth=2.0)
            ax1.plot(df['SMA_200'], label="SMA_200", linewidth=2.0)
            ax2.plot(df['MACDs_8_21_9'], label="MACD_S")
            ax2.plot(df['MACDh_8_21_9'], label="MACD_F")
            ax3.plot(df['RSI_14'], label="RSI_14")
            # ax3.plot(df['CCI_14_0.015'], label="CCI_14")
            ax4.plot(df['OBV'], label="OBV")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper left")
            ax3.legend(loc="upper left")
            ax4.legend(loc="upper left")
            # manager = plt.get_current_fig_manager()
            # manager.resize(*manager.window.maxsize())
            if save:
                plt.savefig(f"data/stocks/plots/TA/{self.id}.png", bbox_inches='tight', dpi=150)
            else:
                plt.show()
        except:
            print(f"Failed to print crypto {self.id}")


    def linear_regression(self):
        self.df.ta.linear_regression()

    @staticmethod
    def print_random(self):
        """
        prints a random crypto
        """
        all_stocks = self.fetch_all_names()
        print(all_stocks[random.randint(2, len(all_stocks) - 1)])


def fetch_all_names():
    all_stocks = []
    # with open('data/sp500.csv') as csv_file:
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
            crypto = CryptoData(name, full=False)
            id, score, rsi, cci = crypto.calc_if_riser_or_faller()
            print(f"id: {id}, score:{score}, rsi:{rsi}, cci:{cci}")
            if score >= FALLER_THRESHOLD:
                fallers[id] = {"score": score, "rsi": rsi, "cci": cci}
            if score <= RISER_THRESHOLD:
                risers[id] = {"score": score, "rsi": rsi, "cci": cci}
        except:
            print("Error 1: Something happened for crypto: ", name)
    util.write('data/stocks/risers/risers.json', risers)
    util.write('data/stocks/fallers/fallers.json', fallers)

    return risers, fallers

if __name__ == '__main__':
    # calc_all_risers_and_fallers()  # this populates the risers and fallers list
    crypto = CryptoData("AGLE", full=True, plot=True)
    # crypto.populate_df_with_indicators()
    # calc_all_risers_and_fallers()
    # crypto.print_df()
    crypto.plot_df(save=True)
    print("success")
