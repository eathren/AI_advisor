import json
import math
from datetime import date, timedelta

import handle_json
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense  # Deep learning classes for recurrent and regular densely-connected layers
from keras.models import Sequential  # Deep learning library, used for neural networks
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from stock_data import StockData
from tensorflow import keras
from tensorflow.keras import layers

"""
This file takes a range of stock closing prices, and then 
trains a model to make an estimate for the stock for the next day

helpful docs:
https://blog.quantinsti.com/neural-network-python/
"""


class NeuralNet:
    def __init__(self, id):
        self.id = id
        self.data = StockData(id, full=True).df
        self.previous_price = 0
        self.predicted_price = 0

    def train(self):
        today = date.today()
        date_today = today.strftime("%Y-%m-%d")
        dataset = self.data
        # Feature Selection - only adjusted close data
        train_df = dataset.filter(['adjusted close'])
        data_unscaled = train_df.values

        # Get the number of rows to train the model on 95% of the data
        train_data_length = math.ceil(len(data_unscaled) * 0.95)

        # Transform features by scaling each feature to a range between 0 and 1
        mm_scaler = MinMaxScaler(feature_range=(0, 1))
        np_data = mm_scaler.fit_transform(data_unscaled)

        # Set the sequence length - this is the timeframe used to make a single prediction
        sequence_length = 50

        # Prediction Index
        index_close = train_df.columns.get_loc("adjusted close")

        # Get the length of the train data set.
        train_data_len = math.ceil(np_data.shape[0] * 0.95)

        # Create the training and test data
        train_data = np_data[0:train_data_len, :]
        test_data = np_data[train_data_len - sequence_length:, :]

        # Generate training data and test data
        x_train, y_train = self.partition_dataset(sequence_length, train_data, index_close)
        x_test, y_test = self.partition_dataset(sequence_length, test_data, index_close)

        # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
        print(x_train.shape, y_train.shape)
        print(x_test.shape, y_test.shape)

        # Validate that the prediction value and the input match up
        # The last close price of the second input sample should equal the first prediction value
        print(x_test[1][sequence_length - 1][index_close])
        print(y_test[0])

        # Configure the neural network model
        model = Sequential()

        neurons = sequence_length

        # Model with sequence_length Neurons
        model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(neurons, return_sequences=False))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=16, epochs=25)

        # Get the predicted values
        y_pred_scaled = model.predict(x_test)
        y_pred = mm_scaler.inverse_transform(y_pred_scaled)
        y_test_unscaled = mm_scaler.inverse_transform(y_test.reshape(-1, 1))

        # Mean Absolute Error (mae)
        mae = mean_absolute_error(y_test_unscaled, y_pred)
        print(f'Median Absolute Error (mae): {np.round(mae, 2)}')

        # Mean Absolute Percentage Error (mape)
        mape = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        print(f'Mean Absolute Percentage Error (mape): {np.round(mape, 2)} %')

        # Median Absolute Percentage Error (mdape)
        mdape = np.median((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        print(f'Median Absolute Percentage Error (mdape): {np.round(mdape, 2)} %')

        # The date from which on the date is displayed
        display_start_date = "2019-01-01"

        # Add the difference between the valid and predicted prices
        train = train_df[:train_data_length + 1]
        valid = train_df[train_data_length:]
        valid.insert(1, "Predictions", y_pred, True)
        valid.insert(1, "Difference", valid["Predictions"] - valid["adjusted close"], True)

        # Zoom in to a closer timeframe
        valid = valid[valid.index > display_start_date]
        train = train[train.index > display_start_date]

        # Visualize the data
        # fig, ax = plt.subplots(figsize=(16, 8), sharex=True) #HERE. This might cause issues.
        fig, ax = plt.subplots()
        plt.title("Predictions vs Ground Truth", fontsize=20)
        plt.rcParams["figure.autolayout"] = True
        plt.xticks(rotation=90)
        plt.ylabel(self.id, fontsize=18)
        plt.plot(train["adjusted close"], color="#039dfc", linewidth=1.0)
        plt.plot(valid["Predictions"], color="#E91D9E", linewidth=1.0)
        plt.plot(valid["adjusted close"], color="black", linewidth=1.0)
        my_xticks = ax.get_xticks()
        plt.xticks([my_xticks[0], my_xticks[len(my_xticks)//2],  my_xticks[-1]], visible=True, rotation="horizontal")
        # Fill between plotlines
        # ax.fill_between(.ytindex, 0, yt["adjusted close"], color="#b9e1fa")
        # ax.fill_between(yv.index, 0, yv["Predictions"], color="#F0845C")
        # ax.fill_between(yv.index, yv["adjusted close"], yv["Predictions"], color="grey")

        # Create the bar plot with the differences
        valid.loc[valid["Difference"] >= 0, 'diff_color'] = "#2BC97A"
        valid.loc[valid["Difference"] < 0, 'diff_color'] = "#C92B2B"
        plt.bar(valid.index, valid["Difference"], width=0.8, color=valid['diff_color'])

        plt.savefig(f"data/stocks/plots/{self.id}.png",bbox_inches='tight', dpi=150)

        # plt.show()
        # Get fresh data
        df_new = dataset.filter(['adjusted close'])

        # Get the last N day closing price values and scale the data to be values between 0 and 1
        last_days_scaled = mm_scaler.transform(df_new[-sequence_length:].values)

        # Create an empty list and Append past n days
        X_test = []
        X_test.append(last_days_scaled)

        # Convert the X_test data set to a numpy array and reshape the data
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Get the predicted scaled price, undo the scaling and output the predictions
        pred_price = model.predict(X_test)
        pred_price_unscaled = mm_scaler.inverse_transform(pred_price)

        # Print last price and predicted price for the next day
        previous_price = round(df_new['adjusted close'][-1], 2)
        predicted_price = round(pred_price_unscaled.ravel()[0], 2)
        percent = round(100 - (predicted_price * 100) / previous_price, 2)

        plus = '+'
        minus = ''
        print(f'The close price for {self.id} at the previous close was {previous_price}')
        print(f'The predicted close price is {predicted_price} ({plus if percent > 0 else minus}{percent}%)')

        # set $ values to be retrieved later in calculate_stocks_daily.py
        self.previous_price = round(previous_price, 2)
        self.predicted_price = round(predicted_price, 2)

    def get_previous_price(self) -> float:
        """
        gets previous price
        :return self.previous_price:
        """
        return self.previous_price

    def get_predicted_price(self) -> float:
        """
        getter for predicted_price
        :return predicted_price:
        """
        return self.predicted_price

    def get_id(self) -> str:
        """
        getter for self.id

        :return self.id:
        """
        return str(self.id)

    def partition_dataset(self, sequence_length, train_df, index_close):
        """
        name: partition_dataset
        This function chunks the dataset and returns that chunk into the sequence lenghts

        :param sequence_length:
        :param train_df:
        :param index_close:
        :return two numpy arrays:
        """
        x, y = [], []
        data_len = train_df.shape[0]
        for i in range(sequence_length, data_len):
            x.append(train_df[i - sequence_length:i, :])  # contains sequence_length values 0-sequence_length * columsn
            y.append(train_df[
                         i, index_close])  # contains the prediction values for validation (3rd column = Close),  for single-step prediction
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y


if __name__ == "__main__":
    """
    Runs the net with a given stock, prints pred and previous price 
    and plots the chart with train data.
    """
    net = NeuralNet("AMC")
    net.train()  # run the net with whatever stock is supplied
