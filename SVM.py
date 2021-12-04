from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from fetch_crypto_data_indicators import CryptoData
import numpy as np
import pandas as pd

class SVM:

    def __init__(self, data):
        self.df = data.df.copy()

        '''
        self.df = self.df.iloc[-100:, :]
        self.df = self.df.reset_index(drop=True)
        '''

        del self.df['high']
        del self.df['low']
        del self.df['volume']
        del self.df['open']
        del self.df['close']
        del self.df['HL']
        del self.df['OC']
        del self.df['5_Day_SMA']
        del self.df['5_Day_EMA']
        del self.df['10_Day_SMA']
        del self.df['10_Day_EMA']
        del self.df['15_Day_EMA']
        del self.df['15_Day_SMA']


        self.add_rise_down_result()
        self.shift_days = data.shift_days
        self.raw_df_for_prediction = data.df_with_na.copy()

        del self.raw_df_for_prediction['high']
        del self.raw_df_for_prediction['low']
        del self.raw_df_for_prediction['volume']
        del self.raw_df_for_prediction['open']
        del self.raw_df_for_prediction['close']
        del self.raw_df_for_prediction['HL']
        del self.raw_df_for_prediction['OC']
        del self.raw_df_for_prediction['5_Day_SMA']
        del self.raw_df_for_prediction['5_Day_EMA']
        del self.raw_df_for_prediction['10_Day_SMA']
        del self.raw_df_for_prediction['10_Day_EMA']
        del self.raw_df_for_prediction['15_Day_EMA']
        del self.raw_df_for_prediction['15_Day_SMA']


        self.last_date = self.raw_df_for_prediction.iloc[-1:, 0:1].values
        self.raw_df = self.df.copy()

        print(self.df)

        normalization_columns = ['adjusted close','next adjusted close',"5_Day_RSI", "10_Day_RSI", "15_Day_RSI", "30_Day_RSI", "60_Day_RSI"]
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(self.df[normalization_columns])

        self.df[normalization_columns] = pd.DataFrame(scaled)

        self.df = self.df.dropna()

        self.df.to_csv("test.csv")

        corr = self.df.corr()

        print(corr)

        self.num_rows, self.num_columns = self.df.shape

        self.split_dataset()
        self.fit()
        self.get_Summary()

        self.predict_test_Set()


    def add_rise_down_result(self):
        self.df['1: rise; 0: down'] = np.where(((self.df['next adjusted close'] - self.df['adjusted close'])/ self.df['adjusted close'] )>= 0.05, 1, 0)
        title = ['date','1: rise; 0: down','adjusted close', 'next adjusted close', "5_Day_RSI", "10_Day_RSI", "15_Day_RSI", "30_Day_RSI", "60_Day_RSI"]
        self.df = self.df[title]

        #self.df.to_csv("test.csv")  # TODO


    def split_dataset(self, TrainSet_percent = 0.9):

        df = self.df
        self.num_train = int(self.num_rows * TrainSet_percent)
        self.num_test = self.num_rows - self.num_train

        num_test = self.num_test
        num_train = self.num_train

        self.train_y = df.iloc[0:num_train, 1]
        self.test_y = df.iloc[num_train:, 1]

        self.train_x = df.iloc[0:num_train, 4:]
        self.test_x = df.iloc[num_train:, 4:]

        self.raw_train_y = self.raw_df.iloc[0:num_train, 1]
        self.raw_test_y = self.raw_df.iloc[num_train:, 1]

        self.raw_train_x = self.raw_df.iloc[0:num_train, 4:]
        self.raw_test_x = self.raw_df.iloc[num_train:, 4:]

        self.num_feature = self.train_x.shape[1]

        self.cofficient = np.zeros(self.num_feature)


    def fit(self):

        alpha = 0.01
        self.lambda_value = 0.1
        iterations = 100
        i = 0
        train_x = self.train_x.to_numpy()
        train_y = self.train_y.to_numpy()
        self.b = 0

        while i < iterations:
            for k, train_x_values in enumerate(train_x):
                if train_y[k] * (np.dot(train_x_values, self.cofficient) - self.b) >= 1:
                    self.cofficient = self.cofficient - alpha * (2 * self.lambda_value * self.cofficient)

                else:
                    self.cofficient = self.cofficient - alpha * (2 * self.lambda_value * self.cofficient - np.dot(train_x_values, train_y[k]))
                    self.b = self.b - alpha * train_y[k]

            i += 1

        '''
        overfit = False
        max_score = -99999999999
        
        while alpha <= 0.1:
            while overfit == False:
                temp_w = self.cofficient
                temp_b = self.b
                i = 0
                while i < iterations:
                    for k, train_x_values in enumerate(train_x):
                        if train_y[k] * (np.dot(train_x_values, temp_w) - temp_b)  >= 1:
                            temp_w = temp_w - alpha * (2 * self.lambda_value * temp_w)

                        else:
                            temp_w = temp_w -  alpha * (2 * self.lambda_value * temp_w - np.dot(train_x_values, train_y[k]))
                            temp_b = temp_b - alpha * train_y[k]

                    i += 1

                prediction = self.raw_test_x.dot(temp_w) - temp_b
                prediction = np.rint(prediction)
                accuracy_df = np.where(self.raw_test_y == prediction, True, False)
                num_true = np.count_nonzero(accuracy_df == True)
                num_test = self.raw_test_y.shape[0]
                score = num_true/num_test


                if score > max_score:
                    max_score = score

                    if i <= 100:
                        iterations += 10
                        self.cofficient = temp_w
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
            alpha += 0.01
            '''


    def predict_test_Set(self):
        prediction = self.test_x.dot(self.cofficient) - self.b
        #prediction = np.rint(prediction)

        prediction = [1 if i > 0.5 else 0 for i in prediction]
        dict = {"prediction": prediction}
        prediction = pd.DataFrame(dict)

        return prediction

    def get_Summary(self):
        prediction = self.predict_test_Set()
        true_list = []
        true_positive_list = []
        true_negative_list = []
        false_positive_list = []
        false_negative_list = []


        for i in range (0, prediction.shape[0]):

            if self.test_y.iloc[0] == prediction.iloc[i, 0]:
                true_list.append(True)
            else:
                true_list.append(False)

            if self.test_y.iloc[0] == 1 and prediction.iloc[i, 0] == 1:
                true_positive_list.append(True)
            else:
                true_positive_list.append(False)

            if self.test_y.iloc[0] == 0 and prediction.iloc[i, 0] == 0:
                true_negative_list.append(True)
            else:
                true_negative_list.append(False)

            if self.test_y.iloc[0] == 0 and prediction.iloc[i, 0] == 1:
                false_positive_list.append(True)
            else:
                false_positive_list.append(False)

            if self.test_y.iloc[0] == 1 and prediction.iloc[i, 0] == 0:
                false_negative_list.append(True)
            else:
                false_negative_list.append(False)

        num_true = true_list.count(True)
        num_true_positive = true_positive_list.count(True)
        num_true_negative = true_negative_list.count(True)
        num_false_positive = false_positive_list.count(True)
        num_false_negative = false_negative_list.count(True)
        num_test = self.test_y.shape[0]
        score = num_true / num_test
        precision = num_true_positive/(num_true_positive + num_false_positive)
        #recall = num_true_positive / (num_true_positive + num_false_negative)

        # for debug
        num_1 = np.count_nonzero(self.test_y == 1)
        print("number of positive(1) in test set", num_1)
        num_0 = np.count_nonzero(self.test_y == 0)
        print("number of negative(0) in test set", num_0)
        pred_to_1 = np.count_nonzero(prediction == 1)
        print("number of predicting to 1 in test set", pred_to_1)
        pred_to_0 = np.count_nonzero(prediction == 0)
        print("number of predicting to 0 in test set", pred_to_0)

        print("\n")

        print("Number of True Prediction: ", num_true)
        print("Number of True Positive Prediction: ", num_true_positive)
        print("Number of True Negative Prediction: ", num_true_negative)
        print("Number of False Positive Prediction: ", num_false_positive)
        print("Number of False Negative Prediction: ", num_false_negative)
        print("Nunber of dataset: ", num_test)
        print("accuracy score: ", score)
        print("Precision is: ", precision)
        #print("Recall: ", recall)


    def predict(self):
        import datetime

        temp = self.raw_df_for_prediction.iloc[-self.shift_days:, 3:]

        prediction = pd.DataFrame()
        prediction['1: rise; 0: down'] = temp.dot(self.cofficient) - self.b

        prediction_result_list = []
        for j in range (0, len(prediction['1: rise; 0: down'])):
            prediction_result_list.append(int(prediction.iloc[j, 0]))

        next_date_list = []
        year = int(self.last_date[0][0][0:4])
        month = int(self.last_date[0][0][5:7])
        day = int(self.last_date[0][0][8:])
        current_date = datetime.date(year, month, day)

        for i in range(0, self.shift_days):

            next_date = current_date + datetime.timedelta(days=1)

            if next_date.weekday() > 4:
                next_date = next_date + datetime.timedelta(days=2)

            next_date_list.append(str(next_date))
            current_date = next_date

        dict = {'date': next_date_list, "rise or down prediction": prediction_result_list}
        result_df = pd.DataFrame(dict)

        return result_df



if __name__ == '__main__':
    data = CryptoData('AMZN')

    SVM_model = SVM(data)

    result = SVM_model.predict()

    print(result)
    '''
    prediction = SVM_model.predict()
    print(ten_day_prediction)
    '''