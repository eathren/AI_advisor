import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import handle_json
from fetch_stock_data import StockData

"""
take the data from one stock
feed it into the nodes


WHAT WE WANT:
Closing stock price of the next day

helpful docs:
https://blog.quantinsti.com/neural-network-python/
"""


def evaluation_function(actual, predicted):  # This might not work
    c = np.sum(.5 * ((actual - predicted) ** 2))


class NeuralNet:
    def __init__(self, id):
        # If file exists, use those values.
        # Else, populate as defaults. Save the file every so often.
        file_path = "data/stocks/neural_nets/" + id + ".json"
        if handle_json.file_exists(file_path):
            print("YES")
            # self.data = pd.read_json()
            # print(self.data)
        else:
            self.data = StockData(id)
            print("NO")
            #     If historical data does not exist, populate neural net with defaults.

        self.learning_rate = 0
        self.historical_success = 0
        self.mentions = 0
        self.curr_trajectory = 0
        self.cost_per_share = 0  # cost/share
        self.PE = 0  # P/E
        self.age = 0
        self.recently_acquired = False

        self.MACD = 0
        self.trend = 0
        self.stoch_rsi = 0
        self.rsi = 0
        self.MA = 0  # Moving average

        self.outputs = 0


if __name__ == "__main__":
    net = NeuralNet("AMZN")
