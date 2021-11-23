import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import handle_json
import json
from fetch_stock_data import StockData
from fetch_crypto_data_indicators import CryptoData

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
        file_path = f"data/stocks/neural_nets/{id}.json"
        self.input = StockData(id).df
        self.input_layer_size = 6
        self.output_layer_size = 1
        self.hidden_layer_size = 3
        # if handle_json.file_exists(file_path):
        #     self.weights = pd.read_json()
        #     # self.data = pd.read_json()
        #     # print(self.data)
        # else:
        self.weights = [.5, .5, .5]
        self.weight = 1.0

        self.data = StockData(id)
        # with open(file_path, "w+") as f:
        #     json.dump(self.input, f, ensure_ascii=False, indent=4)
            #     If historical data does not exist, populate neural net with defaults.
        self.alpha = .01

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

    def activation_sigmoid(self, s):


    def train_all(self, input, goal, epochs):
        pred = input * self.weight
        delta = pred - goal
        error = delta ** 2
        derivative = delta * input
        self.weight = self.weight - (self.alpha * derivative)
        print("Error: " + str(error))

    def train(self):
        for i, row in self.input.iterrows():
            print(i, row['close'], row['open'])

    def predict(self, input):
        return input * self.weight

if __name__ == "__main__":
    net = NeuralNet("AMZN")
    net.train()
