import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import handle_json
import json
from datetime import date, timedelta # Date Functions
import matplotlib.pyplot as plt # For visualization
import matplotlib.dates as mdates # Formatting dates
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error # For measuring model performance / errors
from sklearn.preprocessing import MinMaxScaler #to normalize the price data
from keras.models import Sequential # Deep learning library, used for neural networks
from keras.layers import LSTM, Dense # Deep learning classes for recurrent and regular densely-connected layers

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
        self.id = id
        self.data = StockData(id).df
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
        pass

    def train_all(self, input, goal, epochs):
        pred = input * self.weight
        delta = pred - goal
        error = delta ** 2
        derivative = delta * input
        self.weight = self.weight - (self.alpha * derivative)
        print("Error: " + str(error))

    def train(self):
        # Dimensions of data
        today = date.today()
        date_today = today.strftime("%Y-%m-%d")
        dataset = self.data.df
        # Get fresh data until today
        # Feature Selection - Only Close Data
        train_df = dataset.filter(['adjusted close'])
        data_unscaled = train_df.values

        # Get the number of rows to train the model on 95% of the data
        train_data_length = math.ceil(len(data_unscaled) * 0.95)

        # Transform features by scaling each feature to a range between 0 and 1
        mmscaler = MinMaxScaler(feature_range=(0, 1))
        np_data = mmscaler.fit_transform(data_unscaled)

        # Set the sequence length - this is the timeframe used to make a single prediction
        sequence_length = 50

        # Prediction Index
        index_close = train_df.columns.get_loc("adjusted close")
        # Split the training data into train and train data sets
        # As a first step, we get the number of rows to train the model on 95% of the data
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
        # inputshape = sequence_length Timestamps
        model.add(LSTM(neurons, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(neurons, return_sequences=False))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=16, epochs=25)

        # Get the predicted values
        y_pred_scaled = model.predict(x_test)
        y_pred = mmscaler.inverse_transform(y_pred_scaled)
        y_test_unscaled = mmscaler.inverse_transform(y_test.reshape(-1, 1))
        # Mean Absolute Error (MAE)
        MAE = mean_absolute_error(y_test_unscaled, y_pred)
        print(f'Median Absolute Error (MAE): {np.round(MAE, 2)}')

        # Mean Absolute Percentage Error (MAPE)
        MAPE = np.mean((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        print(f'Mean Absolute Percentage Error (MAPE): {np.round(MAPE, 2)} %')

        # Median Absolute Percentage Error (MDAPE)
        MDAPE = np.median((np.abs(np.subtract(y_test_unscaled, y_pred) / y_test_unscaled))) * 100
        print(f'Median Absolute Percentage Error (MDAPE): {np.round(MDAPE, 2)} %')

        # The date from which on the date is displayed
        display_start_date = "2018-01-01"

        # Add the difference between the valid and predicted prices
        train = train_df[:train_data_length + 1]
        valid = train_df[train_data_length:]
        valid.insert(1, "Predictions", y_pred, True)
        valid.insert(1, "Difference", valid["Predictions"] - valid["adjusted close"], True)

        # Zoom in to a closer timeframe
        valid = valid[valid.index > display_start_date]
        train = train[train.index > display_start_date]

        # Visualize the data
        fig, ax = plt.subplots(figsize=(16, 8), sharex=True)

        plt.title("Predictions vs Ground Truth", fontsize=20)
        plt.ylabel(self.id, fontsize=18)
        plt.plot(train["adjusted close"], color="#039dfc", linewidth=1.0)
        plt.plot(valid["Predictions"], color="#E91D9E", linewidth=1.0)
        plt.plot(valid["adjusted close"], color="black", linewidth=1.0)
        plt.legend(["Train", "Test Predictions", "Ground Truth"], loc="upper left")

        # Fill between plotlines
        # ax.fill_between(yt.index, 0, yt["adjusted close"], color="#b9e1fa")
        # ax.fill_between(yv.index, 0, yv["Predictions"], color="#F0845C")
        # ax.fill_between(yv.index, yv["adjusted close"], yv["Predictions"], color="grey")

        # Create the bar plot with the differences
        valid.loc[valid["Difference"] >= 0, 'diff_color'] = "#2BC97A"
        valid.loc[valid["Difference"] < 0, 'diff_color'] = "#C92B2B"
        plt.bar(valid.index, valid["Difference"], width=0.8, color=valid['diff_color'])

        plt.show()

        # Get fresh data
        df_new = dataset.filter(['adjusted close'])

        # Get the last N day closing price values and scale the data to be values between 0 and 1
        last_days_scaled = mmscaler.transform(df_new[-sequence_length:].values)

        # Create an empty list and Append past n days
        X_test = []
        X_test.append(last_days_scaled)

        # Convert the X_test data set to a numpy array and reshape the data
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Get the predicted scaled price, undo the scaling and output the predictions
        pred_price = model.predict(X_test)
        pred_price_unscaled = mmscaler.inverse_transform(pred_price)

        # Print last price and predicted price for the next day
        price_today = round(df_new['adjusted close'][-1], 2)
        predicted_price = round(pred_price_unscaled.ravel()[0], 2)
        percent = round(100 - (predicted_price * 100) / price_today, 2)

        plus = '+';
        minus = ''
        print(f'The close price for {id} at {today} was {price_today}')
        print(f'The predicted close price is {predicted_price} ({plus if percent > 0 else minus}{percent}%)')

        # print(data)
        # Plotting the graph of returns
        # plt.figure(figsize=(10, 5))
        # plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
        # plt.plot(data['adjusted close'], color='r', label='Market Returns')
        # plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
        # plt.plot(data['volume'], color='g', label='Strategy Returns')
        # plt.legend()
        # plt.show()

    def partition_dataset(self, sequence_length, train_df, index_close):
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

    def predict(self, input):
        return input * self.weight

if __name__ == "__main__":
    net = NeuralNet("INTU")
    net.train()
