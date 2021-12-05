from matplotlib import pyplot as plt
from fetch_crypto_data_indicators import CryptoData
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

class Linear_Regression:

    def __init__(self, data):
        self.df = data.df.copy()

        del self.df['high']
        del self.df['low']
        del self.df['volume']
        del self.df['open']
        del self.df['close']
        del self.df['HL']
        del self.df['OC']
        del self.df['30_Day_RSI']
        del self.df['60_Day_RSI']

        self.raw_df_for_prediction = data.df_with_na.copy()
        del self.raw_df_for_prediction['high']
        del self.raw_df_for_prediction['low']
        del self.raw_df_for_prediction['volume']
        del self.raw_df_for_prediction['open']
        del self.raw_df_for_prediction['close']
        del self.raw_df_for_prediction['HL']
        del self.raw_df_for_prediction['OC']
        del self.raw_df_for_prediction['30_Day_RSI']
        del self.raw_df_for_prediction['60_Day_RSI']

        self.last_date = self.raw_df_for_prediction.iloc[-1:, 0:1].values

        self.raw_df = self.df.copy()


        normalization_columns = ['adjusted close','next adjusted close', '5_Day_SMA', '5_Day_EMA',"10_Day_SMA", "10_Day_EMA", "15_Day_EMA","15_Day_SMA",
                                             "5_Day_RSI", "10_Day_RSI", "15_Day_RSI"]

        for i in range(0, len(normalization_columns)):
            self.df[normalization_columns[i]] = (self.df[normalization_columns[i]] - self.df[normalization_columns[i]].mean() )/ self.df[normalization_columns[i]].std()

        self.df = self.df.dropna()

        self.df.to_csv("test.csv")

        corr = self.df.corr()

        print(corr)

        self.num_rows, self.num_columns = self.df.shape

        self.split_dataset()
        self.fit()
        self.predict_test_Set()
        self.get_Summary()

        self.shift_days = data.shift_days


    def feature_selection(self, df):
        corr = df.corr()

        threshold_corr_per = 0.5
        corr_target = abs(corr['next adjusted close'])
        remove_features = (corr_target[corr_target < threshold_corr_per])

        while len(remove_features) >= 6:
            threshold_corr_per -= 0.05
            remove_features = (corr_target[corr_target < threshold_corr_per])

        self.remove_features_list = []

        for feature in remove_features.index:
            self.remove_features_list.append(feature)

        for each_selected_feature in self.remove_features_list:
            del df[each_selected_feature]


    def split_dataset(self, TrainSet_percent = 0.8):

        df = self.df
        self.num_train = int(self.num_rows * TrainSet_percent)
        self.num_test = self.num_rows - self.num_train

        num_test = self.num_test
        num_train = self.num_train

        self.train_y = df.iloc[0:num_train, 2]
        self.test_y = df.iloc[num_train:, 2]

        self.train_x = df.iloc[0:num_train, 3:]
        self.test_x = df.iloc[num_train:, 3:]

        self.raw_train_y = self.raw_df.iloc[0:num_train, 2]
        self.raw_test_y = self.raw_df.iloc[num_train:, 2]

        self.raw_train_x = self.raw_df.iloc[0:num_train, 3:]
        self.raw_test_x = self.raw_df.iloc[num_train:, 3:]

        self.num_feature = self.train_x.shape[1]

        self.cofficient = np.zeros(self.num_feature)

    # refer from: https://cmdlinetips.com/2020/03/linear-regression-using-matrix-multiplication-in-python-using-numpy/
    # refer from:https://satishgunjal.com/multivariate_lr/
    # refer from: https://faun.pub/implementing-multiple-linear-regression-from-scratch-in-python-f5d84d4935bb
    def fit(self):

        alpha = 0.001
        iterations = 10
        i = 0
        train_x = self.train_x.to_numpy()
        train_y = self.train_y.to_numpy()
        overfit = False

        min_MSE = 99999999999

        self.b = 0
        while alpha <= 0.99:
            while overfit == False:
                temp = self.cofficient
                temp_b = self.b
                i = 0
                while i < iterations:

                    pred = train_x.dot(temp) + temp_b
                    train_x_T = train_x.transpose()
                    loss = np.subtract(pred, train_y)

                    weight = train_x_T.dot(loss) / len(train_y)
                    weight_b = np.sum(loss) / len(train_y)

                    temp = temp - alpha * weight
                    temp_b = temp_b - alpha * weight_b

                    i += 1

                error = np.square(np.subtract(self.raw_train_x.dot(temp) + temp_b, self.raw_train_y))
                MSE = np.mean(np.abs(error))

                if MSE < min_MSE:
                    min_MSE = MSE

                    if i <= 100:
                        iterations += 10
                        self.cofficient = temp
                        self.b = temp_b
                    elif i > 100 and i < 500:
                        iterations += 50
                        self.cofficient = temp
                        self.b = temp_b
                    else:
                        overfit = True

                else:
                    overfit = True
            overfit = False
            iterations = 10
            i = 0
            alpha += 0.001


        from numpy.linalg import inv
        X_transpose = train_x.T
        temp_normal_lr_cofficient = inv(X_transpose.dot(train_x)).dot(X_transpose).dot(train_y)

        temp_normal_lr_error = np.square(np.subtract(self.raw_train_x.dot(temp_normal_lr_cofficient), self.raw_train_y))
        temp_normal_lr_MSE = np.mean(np.abs(temp_normal_lr_error))

        temp_normal_lr_RSS = np.sum((np.subtract(self.test_y, self.test_x.dot(temp_normal_lr_cofficient))) ** 2)
        temp_normal_lr_TSS = np.sum((np.subtract(self.test_y, self.test_y.mean())) ** 2)
        temp_normal_lr_R_sqaure = 1 - (temp_normal_lr_RSS / temp_normal_lr_TSS)

        RSS = np.sum((np.subtract(self.test_y, self.test_x.dot(self.cofficient))) ** 2)
        TSS = np.sum((np.subtract(self.test_y, self.test_y.mean())) ** 2)
        R_sqaure = 1 - (RSS / TSS)

        '''
        if temp_normal_lr_MSE < min_MSE and temp_normal_lr_R_sqaure > 0 and temp_normal_lr_R_sqaure > R_sqaure:
            print("Normal Linear Regression applied")
            self.cofficient = temp_normal_lr_cofficient
        else:
            print("Gradient Descent Linear Regression applied")
        '''


    def predict_test_Set(self):
        prediction = self.raw_test_x.dot(self.cofficient) + self.b
        self.visualization(self.raw_test_y, prediction)

        return prediction

    def visualization(self, true_df, predicted_df):

        visualization_df = pd.DataFrame()
        visualization_df['true close price'] = true_df
        visualization_df['predicted close price'] = predicted_df

        plt.plot(self.raw_df.iloc[self.num_train:, 0], visualization_df['true close price'])
        plt.plot(self.raw_df.iloc[self.num_train:, 0], visualization_df['predicted close price'])
        plt.legend(labels = ['true close price','predicted close price'],loc = 'upper left')
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[::30])
        plt.xticks(rotation=45)
        plt.xticks(fontsize=9)


        plt.xlabel("Date")
        plt.ylabel("Close Price (Unit: Dollar)", fontsize=12)
        plt.title("True Close Price and Predicted Close Price of Tesla", fontsize=15)

        plt.show()


    def predict(self, n = 5):

        import datetime
        year = int(self.last_date[0][0][0:4])
        month = int(self.last_date[0][0][5:7])
        day = int(self.last_date[0][0][8:])
        current_date = datetime.date(year, month, day)

        print("Start to predict ", n, " days close price after", current_date)

        for i in range(0, n):

            next_date = current_date + datetime.timedelta(days=1)

            if next_date.weekday() > 4:
                next_date = next_date + datetime.timedelta(days=2)

            next_date_list = [str(next_date)]

            adjusted_close_list = [0]
            next_adjusted_close_list = [0]
            five_Day_SMA_list = [self.get_SMA_moving_average(5)]
            five_Day_EMA_list = [self.get_EMA_moving_average(5)]
            ten_Day_SMA_list = [self.get_SMA_moving_average(10)]
            ten_Day_EMA_list = [self.get_EMA_moving_average(10)]
            fifteen_Day_SMA_list = [self.get_SMA_moving_average(15)]
            fifteen_Day_EMA_list = [self.get_EMA_moving_average(15)]
            five_Day_RSI_list = [self.get_RSI(5)]
            ten_Day_RSI = [self.get_RSI(10)]
            fifiteen_Day_RSI = [self.get_RSI(15)]

            dict = {'date':next_date_list, 'adjusted close': adjusted_close_list,'next adjusted close': next_adjusted_close_list, '5_Day_SMA': five_Day_SMA_list, '5_Day_EMA': five_Day_EMA_list,
                    "10_Day_SMA": ten_Day_SMA_list, "10_Day_EMA": ten_Day_EMA_list, "15_Day_SMA": fifteen_Day_SMA_list,
                                                "15_Day_EMA": fifteen_Day_EMA_list, "5_Day_RSI" : five_Day_RSI_list,
                                                "10_Day_RSI": ten_Day_RSI, "15_Day_RSI": fifiteen_Day_RSI}
            df = pd.DataFrame(dict)

            self.raw_df_for_prediction = pd.concat([self.raw_df_for_prediction, df], names=['date', 'adjusted close','next adjusted close', '5_Day_SMA', '5_Day_EMA',"10_Day_SMA", "10_Day_EMA", "15_Day_SMA",
                                                "15_Day_EMA", "5_Day_RSI", "10_Day_RSI", "15_Day_RSI"], ignore_index=True)

            self.raw_df_for_prediction.reset_index(drop=True)

            prediction_df = self.raw_df_for_prediction.copy()
            df_x = prediction_df.iloc[-self.shift_days, 3:]
            result_df = df_x.dot(self.cofficient) + self.b

            self.raw_df_for_prediction.at[self.raw_df_for_prediction.tail(1).index, "adjusted close"]= result_df
            current_date = next_date



        plt.plot(self.raw_df_for_prediction.iloc[-(n +30):, 0], self.raw_df_for_prediction.iloc[-(n + 30):, 1])
        plt.plot(self.raw_df_for_prediction.iloc[-(n):, 0], self.raw_df_for_prediction.iloc[-(n):, 1])
        plt.legend(labels = ['true close price','predicted close price'],loc = 'upper right')
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks()[::2])
        plt.xticks(rotation=45)
        plt.xticks(fontsize=9)


        plt.xlabel("Date")
        plt.ylabel("Close Price (Unit: Dollar)", fontsize=12)
        plt.title("Predicted Close Price of Tesla from 12/06/2021 to 12/10/2021", fontsize=15)

        plt.show()


        return self.raw_df_for_prediction.iloc[-(n):, 0:2]



    def get_SMA_moving_average(self, interval):

        df = self.raw_df_for_prediction['adjusted close'].rolling(window = interval).mean()

        return df.iloc[-1]

    def get_EMA_moving_average(self, interval):
        df = self.raw_df_for_prediction['adjusted close'].ewm(com = interval, min_periods=0, adjust=False, ignore_na=False).mean()
        return df.iloc[-1]

    def get_RSI(self, interval):

        adjusted_delta = self.raw_df_for_prediction['adjusted close'].diff()

        up = adjusted_delta.clip(lower = 0)
        down = -1 * adjusted_delta.clip(upper = 0)


        ma_up = up.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()
        ma_down = down.ewm(com = interval - 1, min_periods = 0, adjust = False, ignore_na = False).mean()

        RSI = ma_up/ma_down
        df = 100 - (100/(1 + RSI))
        return df.iloc[-1]


    def get_Summary(self):

        prediction = self.raw_test_x.dot(self.cofficient) + self.b

        lr_error = np.square(np.subtract(prediction, self.raw_test_y))
        lr_MSE = np.mean(np.abs(lr_error))
        print("Mean Square Error is:", lr_MSE)

        MAE = np.mean(np.abs(np.subtract(self.raw_test_y, prediction)))
        print("Mean Absolute Error is:", MAE)

        MAPE = np.mean( np.abs(np.subtract(self.raw_test_y, prediction)/ self.raw_test_y ) )
        print("Mean Absolute Percentage Error is:", MAPE)

        RSS = np.sum((np.subtract(self.raw_test_y, self.raw_test_x.dot(self.cofficient) + self.b)) ** 2)
        TSS = np.sum((np.subtract(self.raw_test_y, self.raw_test_y.mean())) ** 2)

        R_sqaure = 1 - (RSS / TSS)

        print("R square is:", R_sqaure)


if __name__ == '__main__':
    data = CryptoData('TSLA')

    linear_r_model = Linear_Regression(data)
    ten_day_prediction = linear_r_model.predict(10)
    print(ten_day_prediction)
