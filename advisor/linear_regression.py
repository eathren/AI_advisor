from matplotlib import pyplot as plt
from stock_data import StockData
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, id=None, full=False):
        self.id = id
        self.df = StockData('id', full=full).df

    def train(self):
        df = self.df
        print(self.df.columns)
        print(self.df)
        X = np.array(df['close'].values.reshape(-1, 1))
        y = np.array(df['SMA_10'].values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.25)

        regr = LinearRegression()

        regr.fit(X_train, y_train)
        print(regr.ceof_)
        print(regr.score(X_test, y_test))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(
            metrics.mean_squared_error(y_test, y_pred)))
        print('R-2 Score:', metrics.r2_score(y_test, y_pred))

        print(x_test)  # Testing data - In Hours

        y_pred = regressor.predict(x_test)  # Predicting the scores

        # Plotting the regression line
        # line = regressor.coef_*X+regressor.intercept_

        # Plotting for the test data
        plt.scatter(X, y)
        plt.plot(X, line)
        plt.show()

    def plot(self):
        plt.xlabel('Date')
        plt.ylabel('Value | USD')
        plt.plot('')
        plt.title(f"{self.id}")


if __name__ == '__main__':
    data = StockData('TSLA')

    log_reg = LinearRegression('TSLA', full=False)
    log_reg.train()
