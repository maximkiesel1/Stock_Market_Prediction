import sys
import os

from sqlalchemy import create_engine, text, inspect
import ssl

import pandas as pd
import numpy as np

import ta

from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow.compat.v1 as tf

import random


def load_data(database_filepath):
    '''
    Loading the data from a sql database and transform it to X, y data for the machine learning model.
    Transform the data in arrays

    INPUT
    database_filepath - (str) path to the sql database

    OUTPUT
    data - (dict) transformed dfs for the stocks
    '''
    # create the engine for the sql database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # create a connection
    conn = engine.connect()

    # find the table names in the db
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    data = {}
    for name in table_names:
        # transform to a executable object for pandas
        sql = text("SELECT * FROM '{}'".format(name))

        # create the dataframe
        df = pd.read_sql(sql, conn)

        # remove hours, minutes, and seconds from the date
        df['date'] = pd.to_datetime(df['date']).dt.date

        # set the date as index
        df = df.set_index('date')

        # positioning for the target variable on the last column position (for easy finding)
        col = df.pop('Adj Close')
        df['Adj Close'] = col

        data[name] = df

    return data


def data_split(df, window_size):
    '''
    Splitting of the datasets in the dictionaries in train, validation, and test (60%, 20%, 20%).
    Transform the date that it can used in the lstm algorithm.
    Z-Score normalization of the data

    INPUT
    df - (dataframe) Stock dataframe
    windows_size - (int) Number of how far back the program should look at the previous data

    OUTPUT
    Xs_train - (dict) Splits training data of the input features
    Xs_val - (dict) Split validation data of the input features
    Xs_test - (dict) Split test data of the input features
    test_mean - Mean of the adjusted close in the X test data
    test_std - Standard deviation of the adjusted close in the X test data
    ys_train - (dict) Split training data of the output feature
    ys_val - (dict) Split validation data of the output feature
    ys_test - (dict) Split test data of the output feature
    test_y_mean - mean of the adjusted close in the y test data
    test_std - Standard deviation of the adjusted close in the y test data
    '''

    array = np.array(df)

    X, y = [], []

    for i in range(len(array) - window_size):
        X.append(array[i:i + window_size])
        y.append(array[i + window_size])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training, validation, and testing sets
    n_samples = len(X)
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # Standardize the features in X_train, X_val, and X_test
    for i in range(X_train.shape[-1]):
        train_mean = np.mean(X_train[:, :, i])
        train_std = np.std(X_train[:, :, i])
        val_mean = np.mean(X_val[:, :, i])
        val_std = np.std(X_val[:, :, i])
        test_mean = np.mean(X_test[:, :, i])
        test_std = np.std(X_test[:, :, i])

        X_train[:, :, i] = (X_train[:, :, i] - train_mean) / train_std
        X_val[:, :, i] = (X_val[:, :, i] - val_mean) / val_std
        X_test[:, :, i] = (X_test[:, :, i] - test_mean) / test_std

        train_y_mean = np.mean(y_train[:, i])
        train_y_std = np.std(y_train[:, i])
        val_y_mean = np.mean(y_val[:, i])
        val_y_std = np.std(y_val[:, i])
        test_y_mean = np.mean(y_test[:, i])
        test_y_std = np.std(y_test[:, i])

        y_train[:, i] = (y_train[:, i] - train_y_mean) / train_y_std
        y_val[:, i] = (y_val[:, i] - val_y_mean) / val_y_std
        y_test[:, i] = (y_test[:, i] - test_y_mean) / test_y_std

    # the last column (adjusted close) of the X/y test mean and std arrays will be returned (for back transformation)
    return X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std


def train_model(data):
    '''
    Create new features (11) for the X dfs and clean potential Nan values.
    Adjusted the amount of y values.

    INPUT
    data - (dict) Stock dfs with the new features

    OUTPUT
    best_model - (dict) Stocks with following items:
    - 'model': Trained model
    - 'mse': Mean Squared Error for the best model
    - 'paramter': List of the parameter of the best model
    '''

    # define empty dict
    best_model = {}

    # define selection of parameters
    params = {
        'window_size': [4, 8, 12, 20],
        'lstm_units': [16, 32, 64, 128],
        'dense_units': [16, 32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1]
    }

    # iterate through the stock names
    for name in data:

        best_model[name] = {}

        # select the arrays
        df = data[name]

        # iterate through
        for i in range(1, 11):

            random_params = {key: random.choice(values) for key, values in params.items()}

            X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std = data_split(
                df, random_params['window_size'])

            # define the model for the stock JNJ
            model = Sequential()
            model.add(InputLayer((random_params['window_size'], X_train.shape[
                -1])))  # the first parameter is always the window size, the second the number of features
            model.add(LSTM(random_params['lstm_units']))
            model.add(Dense(random_params['dense_units'], 'relu'))
            model.add(Dense(X_train.shape[-1], 'linear'))  # output of 17 features
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=random_params['learning_rate']),
                          metrics=[RootMeanSquaredError()])

            # create a model checkpoint for the best model
            cp = ModelCheckpoint('model/keras_model', save_best_only=True, verbose=0)

            # fitting the model
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[cp], verbose=0)

            # load the best model
            model = load_model('model/keras_model')

            # get the best mse from the best model
            best_mse = history.history['val_loss'][np.argmin(history.history['val_loss'])]

            # if this is the first window size, set the accuracy to the current percent difference
            if i == 1:

                best_model[name]['model'] = model
                best_model[name]['mse'] = best_mse
                best_model[name]['paramter'] = random_params

            # otherwise, compare the current accuracy to the previous best accuracy and update if necessary
            else:

                if best_model[name]['mse'] > best_mse:

                    best_model[name]['model'] = model
                    best_model[name]['mse'] = best_mse
                    best_model[name]['paramter'] = random_params

                else:

                    continue

    return best_model


def model_validation(data, best_model):
    '''
    Visualization of the model performance

    INPUT
    data - data - (dict) Stock dfs with the new features
    best_model - (dict) Stocks with following items:
    - 'model': Trained model
    - 'mse': Mean Squared Error for the best model
    - 'paramter': List of the parameter of the best model

    OUTPUT
    None - plotting the visualization between predicted and actual value, inclusive the variation between the tolerance
    '''

    for name in best_model:
        X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std = data_split(
            data[name], best_model[name]['paramter']['window_size'])

        # make prediction
        test_predictions = best_model[name]['model'].predict(np.array(X_test))

        # select the Adjusted Close for prediction (last column)
        test_predictions_ = test_predictions[:, -1].tolist()

        # back transformation for X data  from the normalization
        transfom_X = ((np.array(test_predictions_) * test_std) + test_mean).tolist()

        # select the Adjusted Close for test data (last column)
        y_test_ = y_test[:, -1].tolist()

        # back transformation for y data  from the normalization
        transfom_y = ((np.array(y_test_) * test_y_std) + test_y_mean).tolist()

        # create a dataframe with predicted values and real values
        test_results = pd.DataFrame(data={'Test Predictions': transfom_X, 'Actuals': transfom_y})

        # calculate the difference between prediction and ground truth
        test_results['diff'] = test_results['Test Predictions'] - test_results['Actuals']

        # calculate the percentage difference between prediction and ground truth
        test_results['diff%'] = (test_results['diff'] / test_results['Actuals']) * 100

        # plotting
        print('######################')
        print(name)
        print('######################')

        # show the statistical informations
        print(test_results.describe())

        # plot test and ground truth data
        plt.figure(figsize=(25, 15))
        plt.plot(test_results['Test Predictions'], label='Test Predictions')
        plt.plot(test_results['Actuals'], label='Actuals')
        plt.title('Comparison Adjusted Close: Predicted And Real Values in Test Data Set')
        plt.ylabel('Adjusted Close')
        plt.xlabel('Timeline')
        plt.legend()
        plt.grid()
        plt.show()

        # plot for percentage deviation between predicted and real values in test data
        plt.figure(figsize=(25, 15))
        plt.plot(test_results.index, test_results['diff%'], color='steelblue', label='Percentage Deviation')
        plt.axhline(y=5, color='r', linestyle=':', label='Upper Tolerance')
        plt.axhline(y=-5, color='r', linestyle=':', label='Lower Tolerance')
        plt.grid()
        plt.legend()
        plt.ylabel('Percentage Deviation[%]')
        plt.xlabel('Timeline')
        plt.title('Percentage Deviation For Adjusted Close Between Predicted And Real Values in Test Data Set')
        plt.show()
        print('The mean for the percentage deviation in the test data is {}.'.format(
            abs(np.mean(abs(test_results['diff%'])))))
        print('-----------------------------------------')

    return None


def main():
    if len(sys.argv) == 1:

        # get path database file
        database_filepath = os.getcwd()[:-5] + 'data/cleaned_data.db'

        print('Load Data From SQL Database...\n')
        # load the data with 6 features
        data = load_data(database_filepath)

        print('Load Data Was Successfull!\n')

        print('Training the models...')
        # train the model
        best_models = train_model(data)

        print('Training Was Successfull!\n')

        # show the best model components
        print('This Are The Best Models With The Following Parameters:')
        print(best_models)

        print('Visualization Of The Results...\n')
        # showing the performance of the best model
        model_validation(data, best_models)

        print('Visualization Was Successfull!\n')

    else:
        print('')


if __name__ == '__main__':
    main()