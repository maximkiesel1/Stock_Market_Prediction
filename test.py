def predict_future(best_models, Xs_test, ys_test):
    '''
    This function takes a LSTM model and predicts the future stock prices.
    Prediction goes from today to 7, 14 and 28 Days in the future.

    INPUT:
    best_models - (dict) dictionary with the best models for each stock
    Xs_test - (dict) dictionary with the X_test data for each stock
    ys_test - (dict) dictionary with the y_test data for each stock
    '''

    prediction_list = []
    # iterate over all stocks
    for stock_name in best_models.keys():
        predictions = best_models[stock_name]['model'].predict(np.array(Xs_test[stock_name][-1])).flatten()

        prediction_list.append(predictions)

