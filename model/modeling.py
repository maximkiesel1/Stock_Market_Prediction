import sys
import os
from sqlalchemy import create_engine, text, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import load_model
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

    # create a dictionary for the data
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

        # fill the dict with loaded data
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
    # transform the dataframe to an array
    array = np.array(df)
    # create empty lists for the X and y values
    X, y = [], []
    # create the X and y values for the lstm algorithm
    for i in range(len(array) - window_size):
        X.append(array[i:i + window_size])
        y.append(array[i + window_size])
    # transform the lists to arrays
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

        # iterate through the random parameters
        for i in range(1, 21):

            # select random parameters
            random_params = {key: random.choice(values) for key, values in params.items()}

            # split the data
            X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std = data_split(
                df, random_params['window_size'])

            # define the model for the stock JNJ
            model = Sequential()
            model.add(InputLayer((random_params['window_size'], X_train.shape[
                -1])))  # the first parameter is always the window size, the second the number of features
            model.add(LSTM(random_params['lstm_units']))
            model.add(Dense(random_params['dense_units'], 'relu'))
            model.add(Dense(X_train.shape[-1], 'linear'))
            model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=random_params['learning_rate']),
                          metrics=[RootMeanSquaredError()])

            # create a model checkpoint for the best model
            cp = ModelCheckpoint('keras_model', save_best_only=True, verbose=0)

            # fitting the model
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, callbacks=[cp], verbose=0)

            # load the best model
            model = load_model('keras_model')

            # get the best mse from the best model
            best_mse = history.history['val_loss'][np.argmin(history.history['val_loss'])]

            # if it is the first iteration, save the model
            if i == 1:

                best_model[name]['model'] = model
                best_model[name]['mse'] = best_mse
                best_model[name]['paramter'] = random_params

            # else compare the mse with the best mse
            else:
                # if the mse is better, save the model
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
    # iterate through the stock names
    for name in best_model:
        # split the data
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
        plt.savefig(os.getcwd()+'/visual_validation/validation'+'_'+ name +'.png')

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
        plt.grid()
        plt.savefig(os.getcwd() + '/visual_validation/variation' + '_' + name + '.png')
        print('The mean for the percentage deviation in the test data is {}.'.format(
            abs(np.mean(abs(test_results['diff%'])))))
        print('-----------------------------------------')

    return None


def prediction_lstm(best_model, data):
    """
    Predict the stock prices (Adjusted Close) using the best LSTM model (with back transformation from the normalization)

    INPUT
    best_model - (dict) A dictionary containing the best LSTM model for each stock.
    data - (dict) Stock dfs with the new features

    OUTPUT:
    prediction - (dict) A dictionary containing the name of the stock, old data, predicted data, and selected days.
    """

    # define empty dict
    prediction = {}

    # Iterate through each stock in the best_model dictionary
    for name in best_model:
        # define empty dict
        prediction[name] = {}
        # define empty list
        prediction[name]['prediction'] = []
        # define empty list
        prediction[name]['selected_days'] = []

        # make predictions for the next 28 days
        for i in range(1, 28 +2):
            # if it is the first iteration, get the last row of the test data and back-transform the Adjusted Close value
            if i == 1:

                # Split the data into training, validation, and test sets
                X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std  = data_split \
                    (data[name], best_model[name]['paramter']['window_size'])

                # Get the last row of the test data and back-transform the Adjusted Close value
                array = np.vstack((X_test[-1][1:], y_test[-1]))
                adj_close = array[:, -1]
                prediction[name]['old_data'] = ((adj_close *test_y_std) + test_y_mean).tolist()

            #
            else:

                # Reshape the array and make a prediction
                array_re = np.reshape(array, (1, best_model[name]['paramter']['window_size'], 6))
                pre = best_model[name]['model'].predict(array_re)

                # Get the predicted Adjusted Close value and back-transform it
                adj_close = pre[-1][-1]
                transform = ((adj_close *test_y_std) + test_y_mean)
                prediction[name]['prediction'].append(transform)

                # Update the array with the new predicted value
                array = np.vstack((array[1:], pre))


        # Select three points from the predicted data and add them to the selected_days list
        prediction[name]['selected_days'].append(prediction[name]['prediction'][6])
        prediction[name]['selected_days'].append(prediction[name]['prediction'][13])
        prediction[name]['selected_days'].append(prediction[name]['prediction'][27])

    return prediction


def visualization_prediction(prediction):
    """
    Plot the predicted stock price (Adjusted Close) and selected days (28 days) for a given prediction.

    INPUT:
    prediction - (dict) A dictionary containing the name of the stock, old data, predicted data, and selected days.

    OUTPUT:
    None - plotting the visualization of the old stock data and the new predictions, inclusive a trading recommendation
    """

    # Iterate through each stock in the prediction dictionary
    for name in prediction:

        # Set up the x and y values for the plot
        x = range(1, len(prediction[name]['old_data']) + len(prediction[name]['prediction']) + 1)
        x1 = x[:len(prediction[name]['old_data']) + 1]
        y1 = [y for y in prediction[name]['old_data']]
        x2 = x[len(prediction[name]['old_data']):]
        y2 = prediction[name]['prediction']
        y1.append(y2[0])

        # Plot the two parts of the data separately, with different colors
        print('######################')
        print(name)
        print('######################')
        plt.figure(figsize=(25, 15))
        plt.plot(x1, y1, color='blue', label='Old Stock Data')
        plt.plot(x2, y2, color='red', linestyle='-.', label='Predicted Stock Price')

        # Set up the x and y values for the selected points
        x_points = [7 + len(x1) - 1, 14 + len(x1) - 1, 28 + len(x1) - 1]
        y_points = prediction[name]['selected_days']
        y_points_round = [round(value, 2) for value in y_points]
        xy_pairs = [(x, y) for x, y in zip(x_points, y_points_round)]

        # Plot the selected points as green dots, with text labels
        for point in xy_pairs:
            plt.scatter(point[0], point[1], color='green', s=100)
            plt.text(point[0] + 0.2, point[1] + 0.1, str(point[1]), fontsize=12)

        # Add labels and legend
        plt.xlabel('Days: Window Size And Prediction Horizon')
        plt.ylabel('Adjusted Close')
        plt.title('Prediction: Adjusted Close')
        plt.legend(['Line'], loc='upper right')
        plt.grid()
        plt.legend()

        # save the plot
        plt.savefig(os.getcwd()+'/visual_predict/predict'+'_'+ name +'.png')

        # Print the predicted stock prices for the selected days and a trading recommendation
        print('')
        print('The predicted Adjusted Close for day 7 is: {}'.format(round(prediction[name]['selected_days'][0], 2)))
        print('The predicted Adjusted Close for day 14 is: {}'.format(round(prediction[name]['selected_days'][1], 2)))
        print('The predicted Adjusted Close for day 28 is: {}'.format(round(prediction[name]['selected_days'][0], 2)))

        # Make a trading recommendation based on the predicted stock prices
        if y1[-1] > y2[-1]:
            print('')
            print('Trading Recommendation after {} days: Sell!'.format(len(x2)))
        else:
            print('')
            print('Trading Recommendation after {} days: Hold!'.format(len(x2)))

        print('-----------------------------------------')
        print('')

    return None

def main():

    if len(sys.argv) == 1:

        # get path database file
        database_filepath = os.getcwd()[:-5] + 'data/cleaned_data.db'

        print('Load Data From SQL Database...\n')
        # load the data with 6 features
        data = load_data(database_filepath)

        print('Load Data Was Successfull!\n')

        print('Training The Models...')
        # train the models
        best_models = train_model(data)

        print('Training Was Successfull!\n')
        # show the best model components
        print('This Are The Best Models With The Following Parameters:\n')
        for name in best_models:
            print(name)
            print(best_models[name])

        print('Visualization Of The Training Results...\n')
        # visualize the results
        model_validation(data, best_models)

        print('Visualizations Were Saved in "/visual_validation"!\n')

        print('Predicting...')
        # predict the stock price
        prediction = prediction_lstm(best_models, data)

        print('\nPredicting Was Successfull!\n')

        print('Visualization Of The Predictions...\n')
        # visualize the prediction
        visualization_prediction(prediction)

        print('Visualizations Were Saved in "/visual_predict"!\n')

    else:
        print('Something Went Wrong... Please Try Again!\n')


if __name__ == '__main__':
    main()



