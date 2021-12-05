import json

import numpy as np
import pandas as pd

# External lib, works to get a fear and greed amount. However, we're reversing that. At > 60, sell. At < 40, buy.
import fear_and_greed


class CryptoData:
    def __init__(self, Stock_Abv):
        self.shift_days = 5
        self.data = self.fetch_data(Stock_Abv)
        self.df = self.create_pd_df()

        self.add_HL()
        self.add_OC()
        self.add_SMA_moving_average(5)
        self.add_EMA_moving_average(5)
        self.add_SMA_moving_average(10)
        self.add_EMA_moving_average(10)
        self.add_SMA_moving_average(15)
        self.add_EMA_moving_average(15)
        self.add_RSI(5)
        self.add_RSI(10)
        self.add_RSI(15)
        self.add_RSI(30)
        self.add_RSI(60)

        self.df_with_na = self.df.copy()

        self.df = self.df.dropna()


    def fetch_data(self, Stock_Abv):
        # this is for not blowing up CoinGecko's API. Use for dev.
        with open('data/stocks/data/' + Stock_Abv+ '.json') as old:
            data = json.load(old)
        return data

    def create_pd_df(self):
        daily_data = self.data["Time Series (Daily)"]
        date_list = []
        adjusted_close_list = []
        next_adjusted_close_list = []
        high_list = []
        low_list = []
        volume_list = []
        open_list = []
        close_list = []

        for date_key in daily_data:
            adjusted_close_list.insert(0,(float(daily_data[date_key]['5. adjusted close'])))
            date_list.insert(0,(date_key))
            next_adjusted_close_list.insert(0, (float(daily_data[date_key]['5. adjusted close'])))
            high_list.append(float(daily_data[date_key]["2. high"]))
            low_list.append(float(daily_data[date_key]["3. low"]))
            volume_list.append(int(daily_data[date_key]["6. volume"]))
            open_list.append(float(daily_data[date_key]["1. open"]))
            close_list.append(float(daily_data[date_key]["4. close"]))


        dict = {'date': date_list, 'adjusted close': adjusted_close_list, 'next adjusted close':next_adjusted_close_list,
                'high': high_list, 'low': low_list, 'volume': volume_list, 'open': open_list, 'close': close_list}
        df = pd.DataFrame(dict)
        df['next adjusted close'] = df['next adjusted close'].shift(-self.shift_days)

        return df

    def add_HL(self):
        self.df['HL'] = self.df['high'] - self.df['low']

    def add_OC(self):
        self.df['OC'] = self.df['close'] - self.df['open']


    def add_SMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_SMA'] = self.df['adjusted close'].rolling(window = interval).mean()

    def add_EMA_moving_average(self, interval):
        self.df[str(interval) + '_Day_EMA'] = self.df['adjusted close'].ewm(com = interval, min_periods=0, adjust=False, ignore_na=False).mean()

    def add_RSI(self, interval):

        adjusted_delta = self.df['adjusted close'].diff()

        up = adjusted_delta.clip(lower = 0)
        down = -1 * adjusted_delta.clip(upper = 0)


        ma_up = up.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()
        ma_down = down.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()

        RSI = ma_up/ma_down
        self.df[str(interval) + '_Day_RSI'] = 100 - (100/(1 + RSI))



if __name__ == '__main__':

    # print(data.top_risers)
    #print(data.fear_and_greed)
    # Add data to a dictionary. If it's in the dict 3 times, then print a medium buy signal.
    main()