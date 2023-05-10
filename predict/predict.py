import os
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.getcwd()[:-7] + 'model')
from modeling import best_models, load_data, data_split


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

        prediction[name] = {}

        prediction[name]['prediction'] = []

        prediction[name]['selected_days'] = []

        # make predictions for the next 28 days
        for i in range(1, 28 +2):

            if i == 1:

                # Split the data into training, validation, and test sets
                X_train, X_val, X_test, test_mean, test_std, y_train, y_val, y_test, test_y_mean, test_y_std  = data_split \
                    (data[name], best_model[name]['paramter']['window_size'])

                # Get the last row of the test data and back-transform the Adjusted Close value
                array = np.vstack((X_test[-1][1:], y_test[-1]))
                adj_close = array[:, -1]
                prediction[name]['old_data'] = ((adj_close *test_y_std) + test_y_mean).tolist()

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

        print('Predicting...')
        prediction= prediction_lstm(best_models, data)

        print('Predicting Was Successfull!\n')

        print('Visualization Of The Predictions...\n')
        visualization_prediction(prediction)

        print('Visualization Was Saved in "/visual_predict"!\n')

    else:
        print('Something went wrong... Please try again!')


if __name__ == '__main__':
    main()