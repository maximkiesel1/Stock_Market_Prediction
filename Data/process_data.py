# import libraries
from sqlalchemy import create_engine
import sys
import tkinter as tk
from tkinter import simpledialog
import yfinance as yf
import datetime
import os

# get user input to load data
def get_user_input():
    '''
    Get the user input to define the following:
    - list of stock short names


    INPUT
    None

    Return
    words_list.split(',') - (list) list of stock names (str) (need to be the specific short names)
    start_date - (str)start date for the download of the data. Format:'YYY-MM-DD'
    end_date - (str) end date for the download of the data. Format:'YYY-MM-DD'
    '''

    root = tk.Tk()
    root.withdraw()

    # open input window for the user
    words_list = simpledialog.askstring(title="Stock Name List (max. 5 Stocks)",
                                        prompt="Please enter a list of stock short names, separated by a space:")

    today = datetime.date.today()

    today_trans = today.strftime("%Y-%m-%d")

    start_date = today - datetime.timedelta(days=365 * 5)

    start_date = start_date.strftime("%Y-%m-%d")

    return words_list.split(' ')[:5], start_date, today_trans


# load data algorithm
def load_stock_data(stock_names, start_date, today_trans):
    '''
    Load specific stock data from yahoo finance api and save them
    in a dataframe.

    INPUT
    stock_names - (list) list of stock names (str) (need to be the specific short names)


    OUTPUT
    dfs - (dict) dictionary with the df+stock names as keys and the matching dataframe as values
    '''


    dfs = {}
    for name in stock_names:
        dfs[name] = yf.download(name, start=start_date, end=today_trans)

    return dfs


# cleaning algorithm
def cleaning_stock_data(stock_df_dict, runs=1):
    '''
    Remove all nan values and outlier from 'Volume'. The outlier will be reduced with the Z-Score. All values
    which are bigger than 3 sigma will be deleted. This happens in different runs to reduce steady the outliers.

    INPUT
    stock_df_dict - (dict) dictionary with the df+stock names as keys and the matching dataframe as values
    runs - (int) number of runs to reduce the number of outlier

    OUTPUT
    clean_stock_df_dict - (dict) cleaned data
    '''

    # create new dict
    clean_stock_df_dict = {}

    # create new dict to count the outlier per run
    number_outlier = {}

    # fill the new dict with keys and 0 as values. This dict sums all outlier for all runs
    count_outlier = {}
    for key in stock_df_dict:
        number_outlier[key] = 10
        count_outlier[key] = 0

    # start with the first run for outlier detection
    for key in stock_df_dict:
        # load the df
        df = stock_df_dict[key]

        # calculate standard deviation and mean
        mean = df['Volume'].mean()
        std = df['Volume'].std()

        # std cutoff value
        threshold = 3

        # calculate z-score
        df['zscore'] = (df['Volume'] - mean) / std

        # detect position of outliers
        outliers = df.loc[abs(df['zscore']) > threshold].index

        count_outlier[key] += len(outliers)

        number_outlier[key] = len(outliers)

        # drop outlier
        df_without_outlier = df.drop(outliers)

        # delete support column 'zscore'
        df_without_outlier.drop('zscore', axis=1, inplace=True)

        # add cleaned df in dictionary
        clean_stock_df_dict[key] = df_without_outlier

        # the following runs for outlier detection
    for run in range(1, runs):
        for key in clean_stock_df_dict:
            if number_outlier[key] > 0:
                # load the df
                df = clean_stock_df_dict[key]

                # calculate standard deviation and mean
                mean = df['Volume'].mean()
                std = df['Volume'].std()

                # std cutoff value
                threshold = 3

                # calculate z-score
                df['zscore'] = (df['Volume'] - mean) / std

                # detect position of outliers
                outliers = df.loc[abs(df['zscore']) > threshold].index

                # update current outlier number
                number_outlier[key] = len(outliers)

                # update the summed outlier number
                count_outlier[key] += len(outliers)

                # drop outlier
                df_without_outlier = df.drop(outliers)

                # delete support column 'zscore'
                df_without_outlier.drop('zscore', axis=1, inplace=True)

                # add cleaned df in dictionary
                clean_stock_df_dict[key] = df_without_outlier

            else:
                continue

    # delete all nan values in the dfs and generate status message about outlier
    # create empty list to store messages
    messages = []
    # delete all nan values in the dfs and generate status message about outlier
    for key in clean_stock_df_dict:
        # select df
        df = clean_stock_df_dict[key]

        # generate status message about NaN values
        messages.append('Here is an overview about the NaN-Values per column for {}:'.format(key))
        messages.append(df.isna().sum())
        messages.append('---------------------------------')

        # drop nan values
        df_without_na = df.dropna()

        # add cleaned df in dictionary
        clean_stock_df_dict[key] = df_without_na
        messages.append('Here is an overview about the outlier for the column "Volume" for {}:'.format(key))
        messages.append('Count Outlier over all runs: {}'.format(count_outlier[key]))
        messages.append('---------------------------------')

    return clean_stock_df_dict, messages


# create a sql database
def save_data(clean_stock_df_dict, database_filepath):
    '''
    Save the dataframe in a sql database

    INPUT
    clean_stock_df_dict - (dict) cleaned dfs in a dictionary
    database_filepath - saving path

    OUTPUT
    None
    '''

    # create sql engine to save the database in a specific file path
    engine = create_engine(f'sqlite:///{database_filepath}')

    # iterate through keys, create dfs and save them
    for key in clean_stock_df_dict:
        df = clean_stock_df_dict[key]

        # change the data from index to column (because the sql can work with it)
        df['date'] = df.index
        df = df.reset_index(drop=True)
        df = df.reindex(columns=['date'] + list(df.columns[:-1]))

        # define table name
        table_name = key

        # transform to the sql database
        df.to_sql(table_name, engine, index=False, if_exists='replace')
    return None

database_filepath = os.getcwd() + '/cleaned_data.db'
def main():
    if len(sys.argv) == 1:

        database_filepath = os.getcwd() + '/cleaned_data.db'

        print('Get user input...\n')
        # select user input
        stock_names, start, end= get_user_input()

        print('Loading data...\n    Stock Names: {}\n    Start Date: {}\n    Today Date: {}'.
              format(stock_names, start, end))

        stock_df_dict = load_stock_data(stock_names, start, end)

        print('Cleaning data...')
        clean_stock_df_dict, messages = cleaning_stock_data(stock_df_dict, runs=1)

        for message in messages:
            print(message)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(clean_stock_df_dict, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the database to save the cleaned data.'
              '\n Example: python3 process_data.py cleaned_data_XXX.db')


if __name__ == '__main__':
    main()
