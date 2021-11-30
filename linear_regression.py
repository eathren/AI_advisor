from fetch_crypto_data_indicators import CryptoData
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Linear_Regression:

    def __init__(self, df):
        self.df = df
        del self.df['date']
        del self.df['high']
        del self.df['low']
        del self.df['volume']

        self.feature_selection()

        self.raw_df = df

        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(self.df.values)

        self.df = pd.DataFrame(scaled)

        self.num_rows, self.num_columns = df.shape


    def feature_selection(self):
        corr = self.df.corr()

        threshold_corr_per = 0.85
        corr_target = abs(corr['adjusted close'])
        remove_features = (corr_target[corr_target < threshold_corr_per])

        while len(remove_features) >= 14:
            threshold_corr_per - 0.05
            remove_features = (corr_target[corr_target < threshold_corr_per])

        remove_features_list = []

        for feature in remove_features.index:
            remove_features_list.append(feature)

        for each_selected_feature in remove_features_list:
            del self.df[each_selected_feature]


    def split_dataset(self, TrainSet_percent):

        df = self.df
        self.num_train = int(self.num_rows * TrainSet_percent)
        self.num_test = self.num_rows - self.num_train

        num_test = self.num_test

        self.train_y = df.iloc[num_test:, 0]
        self.test_y = df.iloc[0:num_test, 0]

        self.train_x = df.iloc[num_test:, 1:]
        self.test_x = df.iloc[0:num_test, 1:]

        self.raw_train_y = self.raw_df.iloc[num_test:, 0]
        self.raw_test_y = self.raw_df.iloc[0:num_test, 0]

        self.raw_train_x = self.raw_df.iloc[num_test:, 1:]
        self.raw_test_x = self.raw_df.iloc[0:num_test, 1:]

        self.num_feature = self.train_x.shape[1]

        self.cofficient = np.zeros(self.num_feature)

    def get_learning_cost(self):
        cost = np.sum((((self.train_x.dot(self.cofficient) + self.b) - self.train_y) ** 2) / (2 * len(self.train_y)))
        return cost

    # refer from: https://cmdlinetips.com/2020/03/linear-regression-using-matrix-multiplication-in-python-using-numpy/
    # refer from:https://satishgunjal.com/multivariate_lr/
    # refer from: https://faun.pub/implementing-multiple-linear-regression-from-scratch-in-python-f5d84d4935bb
    def fit(self, alpha):

        iterations = 10
        i = 0
        train_x = self.train_x.to_numpy()
        train_y = self.train_y.to_numpy()
        overfit = False

        min_MSE = 9999999
        while overfit == False:
            temp = self.cofficient
            while i < iterations:

                pred = train_x.dot(temp)
                train_x_T = train_x.transpose()
                loss = np.subtract(pred, train_y)

                weight = train_x_T.dot(loss) / len(train_y)

                temp = temp - alpha * weight

                i += 1

            error = np.square(np.subtract(self.raw_train_x.dot(temp), self.raw_train_y))
            MSE = np.mean( np.abs(error))

            if MSE < min_MSE:
                min_MSE = MSE
                iterations += 10
                self.cofficient = temp
            else:
                overfit = True



        from numpy.linalg import inv
        X_transpose = train_x.T
        temp_normal_lr_cofficient = inv(X_transpose.dot(train_x)).dot(X_transpose).dot(train_y)

        temp_normal_lr_error = np.square(np.subtract(self.raw_train_x.dot(temp_normal_lr_cofficient), self.raw_train_y))
        temp_normal_lr_MSE = np.mean(np.abs(temp_normal_lr_error))

        if temp_normal_lr_MSE < min_MSE:
            self.cofficient = temp_normal_lr_cofficient


    def predict(self, dataset):
        dataset.to_numpy()
        return dataset.dot(self.cofficient)

    def get_Summary(self):

        lr_error = np.square(np.subtract(self.raw_test_x.dot(self.cofficient), self.raw_test_y))
        lr_MSE = np.mean(np.abs(lr_error))
        print("Mean Square Error is:", lr_MSE)

        MAE = np.mean(np.abs(np.subtract(self.raw_test_y, self.raw_test_x.dot(self.cofficient))))
        print("Meam Absolute Error is:", MAE)

        MPE = np.mean(np.subtract(self.raw_test_y, self.raw_test_x.dot(self.cofficient))/ self.raw_test_y )
        print("Meam Percentage Error is:", MPE)

        MAPE = np.mean( np.abs(np.subtract(self.raw_test_y, self.raw_test_x.dot(self.cofficient))/ self.raw_test_y ) )
        print("Meam Absolute Percentage Error is:", MAPE)



if __name__ == '__main__':
    data = CryptoData('TSLA')

    linear_r_model = Linear_Regression(data.df)
    linear_r_model.split_dataset(0.8)
    linear_r_model.fit(0.01)

    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = data.df.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns)

    plt.show()

    print(corr)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data.df.iloc[:, 1:], data.df.iloc[:, 0:1], test_size=0.2, random_state=0)

    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(y_pred)


    prediction = linear_r_model.predict(linear_r_model.raw_test_x)

    true_y = linear_r_model.raw_test_y

    print(prediction)
    print(true_y)

    linear_r_model.get_Summary()

