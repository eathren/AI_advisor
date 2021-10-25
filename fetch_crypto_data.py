import json

import numpy as np


class CryptoData:
    def __init__(self):
        self.data = self.fetch_data()
        self.top_risers = self.get_weekly_risers()
        self.best_to_buy = []
        self.best_to_sell = []

    # def fetch_data(self):
    #     raw = requests.get(
    #         "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=true")
    #     data = np.array(raw.json())
    # return data

    def fetch_data(self):
        "this is for not blowing up CoinGecko's API. Use for dev."
        with open('data/old.json') as old:
            raw = json.load(old)
        return raw

    def moving_average(self, a, n=7):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

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
    # this is for live data.
    # raw = requests.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=true")
    # data = np.array(raw.json())

    # This is for when using old json data to avoid api abuse.
    data = CryptoData()
    print(data.top_risers)
    # print(data)

    # Add data to a dictionary. If it's in the dict 3 times, then print a medium buy signal.
