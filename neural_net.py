import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# take the data from one stock
# feed it into the nodes


class NeuralNet:
    def __init__(self):
        self.learning_rate = 0
        self.historical_success = 0
        self.mentions = 0
        self.curr_trajectory = 0
        self.cost_per_share = 0 # cost/share
        self.PE = 0 # P/E
        self.age = 0
        self.recently_acquired = False

        self.MACD = 0
        self.trend= 0
        self.stoch_rsi = 0
        self.rsi = 0
        self.MA = 0 # Moving average


        self.outputs = 0

