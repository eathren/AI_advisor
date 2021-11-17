import json

import numpy as np
import pandas as pd

# External lib, works to get a fear and greed amount. However, we're reversing that. At > 60, sell. At < 40, buy.
import fear_and_greed

"""

Functions to add: MSE
target = 0

mse = np.square(prediction - target) //https://en.wikipedia.org/wiki/Mean_squared_error

FOR EACH COIN IN THE TOP 100:
    COMPARE STOCH RSI
    
    COMPARE MACD
        
    BINARY GATES
        IF MACD IS DECREASING AND STOCH RSI > 80: 
            SELL
        IF MACD IS INCREASING AND RSI < 80:
            BUY
        IF MACD IS INCREASING AND RSI > 80:
                
    Quantitative value investing… Predict 6-month price movements based fundamental indicators from companies’ quarterly reports.
    
    Forecasting… Build time series models, or even recurrent neural networks, on the delta between implied and actual volatility.
    
    Statistical arbitrage… Find similar stocks based on their price movements and other factors and look for periods when their prices diverge.
    
"""


class CryptoData:
    def __init__(self):
        self.data = self.fetch_data()
        self.df = self.create_pd_df()


        #self.top_risers = self.get_weekly_risers()
        #self.best_to_buy = []
        #self.best_to_sell = []
        #self.fear_and_greed = fear_and_greed.get().value
        #self.buy_signal = False

    # FOR LIVE DATA USE THIS
    # def fetch_data(self):
    #     raw = requests.get(
    #         "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=true")
    #     data = np.array(raw.json())
    #     return data

    def fetch_data(self):
        # this is for not blowing up CoinGecko's API. Use for dev.
        with open('data/stocks/data/TSLA.json') as old:
            data = json.load(old)
        return data

    def create_pd_df(self):
        daily_data = self.data["Time Series (Daily)"]
        date_list = []
        adjusted_close_list = []
        high_list= []
        low_list = []
        volume_list = []

        for date_key in daily_data:
            adjusted_close_list.append(float(daily_data[date_key]['5. adjusted close']))
            date_list.append(date_key)
            high.append(float(daily_data[date_key]["2. high"]))
            low.append(float(daily_data[date_key]["3. low"]))
            volume.append(int(daily_data[date_key]["6. volume"]))

        dict = {'date': date_list, 'adjusted close': adjusted_close_list,
                                    'high':high_list, 'low':low_list, 'volume': volume_list}

        df = pd.DataFrame(dict)

        return df


    def add_SMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_SMA'] = self.df['adjusted close'].rolling(window = interval).mean()

    def add_EMA_moving_average(self, num_days):
        self.df[str(interval) + '_Day_EMA'] = self.df['adjusted close'].ewm(com = interval, min_periods=0, adjust=False, ignore_na=False).mean()

    def add_RSI(self, interval):

        adjusted_delta = self.df['adjusted close'].diff()

        up_clip = adjusted_delta.clip(lower = 0)
        down_clip = -1 * adjusted_delta.clip(upper = 0)


        ma_up = up_clip.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()
        ma_down = down_clip.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()

        RSI = ma_up/ma_down
        self.df[str(interval) + '_Day_RSI'] = 100 - (100/(1 + RSI))


    def add_stochastic_RSI(self, interval, K = 3, D = 3):
        adjusted_delta = self.df['adjusted close'].diff()

        up_clip = adjusted_delta.clip(lower=0)
        down_clip = -1 * adjusted_delta.clip(upper=0)

        ma_up = up_clip.ewm(com= interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()
        ma_down = down_clip.ewm(com= interval - 1, min_periods=0, adjust=False, ignore_na=False).mean()

        RSI = ma_up / ma_down
        RSI = 100 - (100 / (1 + RSI))

        stochastic_RSI = (RSI - RSI.rolling(interval).min()) / (RSI.rolling(interval).max() - RSI.rolling(interval).min())

        self.df[str(interval) + '_Day_Stochastic_RSI'] = stochastic_RSI
        self.df[str(interval) + '_Day_Stochastic_RSI_K'] = stochastic_RSI.rolling(K).mean()

        stochastic_RSI_K = stochastic_RSI.rolling(K).mean()
        self.df[str(interval) + '_Day_Stochastic_RSI_D'] = stochastic_RSI_K.rolling(D).mean()


    # refer from: https://www.alpharithms.com/calculate-macd-python-272222/
    def add_MACD(self, fast = 12, slow = 26, signal = 9):

        K = self.df['adjusted close'].ewm(span= fast, adjust = False, min_periods = fast).mean()
        D = self.df['adjusted close'].ewm(span = slow, adjust = False, min_periods = slow).mean()

        MACD = K - D
        MACD_S = MACD.ewm(span = signal, adjust = False, min_periods = signal).mean()
        MACD_H = MACD - MACD_S

        self.df['MACD'] = MACD
        self.df['MACD_H'] = MACD_H
        self.df['MACD_S'] = MACD_S

    # refer from https://python.plainenglish.io/trading-using-python-average-directional-index-adx-aeab999cffe7
    def add_ADX(self, num_days):
        return null  # TODO




    def get_weekly_risers(self):
        top_risers = []
        for coin in self.data[0:100]:
            id = coin["id"]
            sparkline = coin["sparkline_in_7d"]["price"]
            moving_avg = self.moving_average(sparkline)
            # moving_avg = (sum(sparkline) / len(sparkline)) / 7
            top_risers.append((id, moving_avg))
        # top_risers = sorted(top_risers, key = lambda t: t[1], reverse=True)
        print(top_risers)
        return top_risers


if __name__ == '__main__':
    data = CryptoData()

    data.add_SMA_moving_average(14) #need to drop 0:interval - 1
    data.add_EMA_moving_average(14) #need to drop 0:interval - 1
    data.add_RSI(14) #need to drop 0:interval - 1
    data.add_stochastic_RSI(14) #need to drop 0:interval - 1
    data.add_MACD() #need to drop 0:slow - 1



    print(data.df)


    #print(data.top_risers)
    #print(data.fear_and_greed)

    # Add data to a dictionary. If it's in the dict 3 times, then print a medium buy signal.
