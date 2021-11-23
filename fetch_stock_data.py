import os
import requests
from dotenv import load_dotenv
import csv
import random
import pandas as pd
import numpy as np
import json
# import yfinance as yf
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
        # self.data = self.read_data()
        self.data = self.fetch_json()
        self.df = self.json_to_pd_df()
        # populate indicators in pandas dataframe data.
        self.add_SMA_moving_average(7)  # need to drop 0:interval - 1
        self.add_EMA_moving_average(7)  # need to drop 0:interval - 1
        self.add_SMA_moving_average(14)  # need to drop 0:interval - 1
        self.add_EMA_moving_average(14)  # need to drop 0:interval - 1
        self.add_SMA_moving_average(21)  # need to drop 0:interval - 1
        self.add_EMA_moving_average(21)  # need to drop 0:interval - 1
        self.add_RSI(14)  # need to drop 0:interval - 1
        self.add_stochastic_RSI(14)  # need to drop 0:interval - 1
        self.add_MACD(12, 21, 9)  # need to drop 0:slow - 1
        self.add_ADX(14)
        self.add_OBV(14)
        self.add_HL()
        self.add_OC()
        # self.add_AD_line(14)
        self.df = self.df.dropna()

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
        params = {'symbol': self.id, 'apikey': AA_KEY, 'outputsize':'full'}
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
        df = pd.DataFrame(self.data['Time Series (Daily)']).transpose()
        # Pandas is dumb when it comes to renaming rows. Make them columns briefly to rename instead.
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
        print(df)
        return df

    def add_SMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_SMA'] = self.df['adjusted close'].rolling(window=interval).mean()

    def add_EMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_EMA'] = self.df['adjusted close'].ewm(com=interval, min_periods=0, adjust=False,
                                                                            ignore_na=False).mean()

    def add_RSI(self, interval):
        adjusted_delta = self.df['adjusted close'].astype(float).diff()
        up = adjusted_delta.clip(lower=0)

        down = -1 * adjusted_delta.clip(upper=0)

        ma_up = up.ewm(com=interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()

        ma_down = down.ewm(com=interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()

        RSI = ma_up / ma_down

        self.df[str(interval) + '_Day_RSI'] = 100 - (100 / (1 + RSI))

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
        print('1')
        for index, row in self.df.iterrows():
            if row['high'] != row['low']:
                accumulation = ((row["adjusted close"] - row['low']) - (row['high'] - row['adjusted close'])) / (
                        row['high'] - row['low']) * row['volume']
            else:
                accumulation = 0

            accumulation_list.append(accumulation)
        self.df['A/D line'] = accumulation_list
        self.df[str(interval) + '_EMA_AD-line'] = self.df['A/D line'].ewm(ignore_na=False, min_periods=0, com=interval)

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


def fetch_fresh_data():
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
        '''
    stock = StockData("AACG")
    fetch_fresh_data()
    print("yes", stock.df)
