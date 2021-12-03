import os
import requests
from dotenv import load_dotenv
import csv
import random
import pandas as pd
import numpy as np
import json
from time import sleep
import pandas_ta as ta

import handle_json
import matplotlib.pyplot as plt

load_dotenv()

AA_KEY = os.getenv('AA_KEY')

RISER_THRESHOLD = -3
FALLER_THRESHOLD = 3

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
    def __init__(self, id=None, full=True, new_data = False):
        self.id = id.upper()
        self.full = full

        if new_data:
            self.data = self.fetch_json()
        else:
            self.data = self.read_data()
        self.df = self.json_to_pd_df()
        #
        # populate indicators in pandas dataframe data.
        self.add_SMA_moving_average(7)
        self.add_EMA_moving_average(7)
        self.add_SMA_moving_average(14)
        self.add_EMA_moving_average(14)
        self.add_SMA_moving_average(21)
        self.add_EMA_moving_average(21)
        self.add_RSI(7)
        self.add_RSI(14)
        self.add_RSI(30)
        self.add_stochastic_RSI(14)
        self.add_MACD(12, 21, 9)  # need to drop 0:slow - 1
        self.add_ADX(14)
        self.add_OBV(14)
        self.add_HL()
        self.add_OC()
        # self.add_AD_line(14)
        # self.df = self.df.dropna()
        self.df = self.df.iloc[::-1]  # order dates in opposite order.

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
        if self.full:
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
                                '5. adjusted close': 'adjusted close', '6. volume': 'volume',
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
            # other choice to dump: self.data.to_json()
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    def create_pd_df(self):
        daily_data = self.data["Time Series (Daily)"]
        date_list = []
        adjusted_close_list = []
        high_list = []
        low_list = []
        volume_list = []

        for date_key in daily_data:
            adjusted_close_list.append(float(daily_data[date_key]['5. adjusted close']))
            date_list.append(date_key)
            high_list.append(float(daily_data[date_key]["2. high"]))
            low_list.append(float(daily_data[date_key]["3. low"]))
            volume_list.append(int(daily_data[date_key]["6. volume"]))

        dict = {'date': date_list, 'adjusted close': adjusted_close_list,
                'high': high_list, 'low': low_list, 'volume': volume_list}

        df = pd.DataFrame(dict)
        return df

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
        rsi = df.ta.rsi(close='adjusted close', length=14, append=True)
        macd = df.ta.macd(close='adjusted close', fast=12, slow=26, signal=9, append=True, signal_indicators=True)
        cci = df.ta.cci(high='high', low='low', close='adjusted close', append=True)

        cci_val = df['CCI_14_0.015'].iloc[-1]

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
        elif rsi_val < 20:  # Rsi value is below 30, buy signal
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
            score += 3
        elif cci_val < -200:
            score += 4

        return self.id, score, rsi_val, cci_val

    def add_SMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_SMA'] = self.df['adjusted close'].rolling(window=interval).mean()

    def add_EMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_EMA'] = self.df['adjusted close'].ewm(com=interval, min_periods=0, adjust=False,
                                                                            ignore_na=False).mean()

    def add_RSI(self, interval):
        delta = self.df['adjusted close'].astype(float).diff()
        up = delta.clip(lower=0)

        down = -1 * delta.clip(upper=0)

        ema_up = up.ewm(com=interval, adjust=False).mean()

        ema_down = down.ewm(com=interval, adjust=False).mean()

        rs = round(ema_up / ema_down, 2)

        ind = str(interval) + '_Day_RSI'

        self.df[ind] = 100 - (100 / (1 + rs))

    def add_stochastic_RSI(self, interval, K=3, D=3):
        adjusted_delta = self.df['adjusted close'].diff()

        up = adjusted_delta.clip(lower=0)
        down = -1 * adjusted_delta.clip(upper=0)

        ma_up = up.ewm(com=interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()
        ma_down = down.ewm(com=interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()

        RSI = ma_up / ma_down
        RSI = 100 - (100 / (1 + RSI))

        stochastic_RSI = (RSI - RSI.rolling(interval).min()) / (
                RSI.rolling(interval).max() - RSI.rolling(interval).min())

        self.df[str(interval) + '_Day_Stochastic_RSI'] = stochastic_RSI
        self.df[str(interval) + '_Day_Stochastic_RSI_K'] = stochastic_RSI.rolling(K).mean()

        stochastic_RSI_K = stochastic_RSI.rolling(K).mean()
        self.df[str(interval) + '_Day_Stochastic_RSI_D'] = stochastic_RSI_K.rolling(D).mean()

    # refer from: https://www.alpharithms.com/calculate-macd-python-272222/
    def add_MACD(self, fast=12, slow=26, signal=9):

        K = self.df['adjusted close'].ewm(span=fast, adjust=False, min_periods=fast).mean()
        D = self.df['adjusted close'].ewm(span=slow, adjust=False, min_periods=slow).mean()

        MACD = K - D
        MACD_S = MACD.ewm(span=signal, adjust=False, min_periods=signal).mean()
        MACD_H = MACD - MACD_S

        self.df['MACD'] = MACD
        self.df['MACD_H'] = MACD_H
        self.df['MACD_S'] = MACD_S

    # refer from https://python.plainenglish.io/trading-using-python-average-directional-index-adx-aeab999cffe7
    def add_ADX(self, interval=14):

        self.df['negative_DM'] = self.df['low'].shift(1) - self.df['low']
        self.df['positive_DM'] = self.df['high'] - self.df['high'].shift(1)

        self.df['positive_DM'] = np.where(
            (self.df['positive_DM'] > self.df['negative_DM']) & (self.df['positive_DM'] > 0), self.df['positive_DM'],
            0.0)

        self.df['negative_DM'] = np.where(
            (self.df['negative_DM'] > self.df['positive_DM']) & (self.df['negative_DM'] > 0), self.df['negative_DM'],
            0.0)

        self.df['true_range_1'] = self.df['high'] - self.df['low']
        self.df['true_range_2'] = np.abs(self.df['high'] - self.df['adjusted close'].shift(1))
        self.df['true_range_3'] = np.abs(self.df['low'] - self.df['adjusted close'].shift(1))

        self.df['true_range'] = self.df[['true_range_1', 'true_range_2', 'true_range_3']].max(axis=1)

        self.df[str(interval) + '_true_range'] = self.df['true_range'].rolling(interval).sum()
        self.df[str(interval) + 'positive_directional_movement_index'] = self.df['positive_DM'].rolling(interval).sum()
        self.df[str(interval) + 'negative_directional_movement_index'] = self.df['negative_DM'].rolling(interval).sum()

        self.df[str(interval) + 'positive_directional_index'] = self.df[
                                                                    str(interval) + 'positive_directional_movement_index'] / \
                                                                self.df[str(interval) + '_true_range'] * 100
        self.df[str(interval) + 'negative_directional_index'] = self.df[
                                                                    str(interval) + 'negative_directional_movement_index'] / \
                                                                self.df[str(interval) + '_true_range'] * 100
        self.df[str(interval) + '_negative_directional_index_temp'] = abs(
            self.df[str(interval) + 'positive_directional_index'] - self.df[
                str(interval) + 'negative_directional_index'])
        self.df[str(interval) + 'directional_index'] = self.df[str(interval) + 'positive_directional_index'] + self.df[
            str(interval) + 'negative_directional_index']

        self.df['Directional_Index'] = 100 * (self.df[str(interval) + '_negative_directional_index_temp'] / self.df[
            str(interval) + 'directional_index'])

        self.df[str(interval) + '_ADX'] = self.df['Directional_Index'].rolling(interval).mean()
        self.df[str(interval) + '_ADX'] = self.df[str(interval) + '_ADX'].fillna(self.df[str(interval) + '_ADX'].mean())
        del self.df['Directional_Index']
        del self.df['true_range_1']
        del self.df['true_range_2']
        del self.df['true_range_3']
        del self.df[str(interval) + 'positive_directional_index']
        del self.df[str(interval) + 'negative_directional_index']
        del self.df[str(interval) + '_negative_directional_index_temp']
        del self.df[str(interval) + 'directional_index']
        del self.df[str(interval) + 'positive_directional_movement_index']
        del self.df[str(interval) + 'negative_directional_movement_index']
        del self.df['positive_DM']
        del self.df['negative_DM']
        del self.df[str(interval) + '_true_range']
        del self.df['true_range']

    # refer from: https://randerson112358.medium.com/stock-trading-strategy-using-on-balance-volume-obv-python-77a7c719cdac
    def add_OBV(self, interval):
        On_balance_Volumn = []
        adjust_close_list = []
        On_balance_Volumn.append(0)

        for index in self.df.index:
            adjust_close_list.append(self.df['adjusted close'][index])

        for i in range(1, len(adjust_close_list)):
            previous_OBV = On_balance_Volumn[-1]
            current_volumn = adjust_close_list[i]
            if adjust_close_list[i] > adjust_close_list[i - 1]:
                On_balance_Volumn.append(previous_OBV + current_volumn)

            elif adjust_close_list[i] < adjust_close_list[i - 1]:
                On_balance_Volumn.append(previous_OBV - current_volumn)
            else:
                On_balance_Volumn.append(previous_OBV)

        self.df['OBV'] = On_balance_Volumn
        self.df[str(interval) + '_EMA_OBV'] = self.df['OBV'].ewm(com=interval).mean()

    def add_HL(self):
        self.df['HL'] = self.df['high'] - self.df['low']

    def add_OC(self):
        self.df['OC'] = self.df['close'] - self.df['open']

    # refer from: https://github.com/voice32/stock_market_indicators/blob/master/indicators.py
    def add_AD_line(self, interval):
        accumulation_list = []
        for index, row in self.df.iterrows():
            if row['high'] != row['low']:
                accumulation = ((row["adjusted close"] - row['low']) - (row['high'] - row['adjusted close'])) / (
                        row['high'] - row['low']) * row['volume']
            else:
                accumulation = 0

            accumulation_list.append(accumulation)
        self.df['A/D line'] = accumulation_list
        self.df[str(interval) + '_EMA_AD-line'] = self.df['A/D line'].ewm(ignore_na=False, min_periods=0, com=interval)

    def print_random(self):
        """
        prints a random stock
        """
        all_stocks = self.fetch_all_names()
        print(all_stocks[random.randint(2, len(all_stocks) - 1)])


def fetch_fresh_data():
    all_stocks = fetch_all_names()
    for i, name in enumerate(all_stocks[1:]):
        sleep(1)  # this should be 1 second, api is limited to 70 req/min
        stock = StockData(name, full=True)
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


def calc_all_risers_and_fallers():
    all_stocks = fetch_all_names()
    risers = {}
    fallers = {}
    for i, name in enumerate(all_stocks[1:50]):
        try:
            stock = StockData(name, full=True, new_data = False)
            id, score, rsi, cci = stock.calc_if_riser_or_faller()
            print(f"id: {id}, score:{score}, rsi:{rsi}, cci:{cci}")
            if score >= FALLER_THRESHOLD:
                fallers[id] = {"score": score, "rsi": rsi, "cci": cci}
            if score <= RISER_THRESHOLD:
                print("YASS")
                risers[id] = {"score": score, "rsi": rsi, "cci":cci}
        except:
            print("Something happend for stock: ", name)
    with open('data/stocks/risers/risers.json', 'w', encoding='utf-8') as f:
        json.dump(risers, f, ensure_ascii=False, indent=4)
    with open('data/stocks/fallers/fallers.json', 'w', encoding='utf-8') as f:
        json.dump(fallers, f, ensure_ascii=False, indent=4)
    return risers, fallers

"""     
N: 1-1000 stocks
Z: 1001 - 2000
K: 2000 - 3000
"""

if __name__ == '__main__':
    # fetch_fresh_data()  # this updates the data every day after closing
    calc_all_risers_and_fallers()  # this populates the risers and fallers list
