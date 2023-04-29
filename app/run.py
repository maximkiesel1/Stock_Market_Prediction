import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
def load_data(database_filepath):
    '''
    Loading the data from a sql database.

    INPUT
    database_filepath - (str) path to the sql database

    OUTPUT
    data - (dict) Features dfs for the stocks
    '''
    # create the engine for the sql database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # create a connection
    conn = engine.connect()

    # find the table names in the db
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    # create empty dict
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

        # insert table in dict
        data[name] = df

    return data


def generate_stock_visualizations(data):

    for stock_name in table_names:
        window_size = 20

        plt.figure(figsize=(25, 15))
        # calculate the rolling average
        rolling_mean_high = data[stock_name]['High'].rolling(window_size).mean()
        rolling_mean_low = data[stock_name]['Low'].rolling(window_size).mean()

        # plot rolling mean variants
        plt.plot(rolling_mean_high, label="Rolling Mean High")
        plt.plot(rolling_mean_low, label="Rolling Mean Low")
        plt.legend(loc='upper left')
        plt.ylabel('Stock Price Rolling Average')
        plt.grid()

        plt.savefig(f'{stock_name}_stock.png')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_name = request.form["stock_name"]
        df = get_stock_data(stock_name)
        generate_stock_visualizations(data)
        return send_file(f'{stock_name}_stock.png', as_attachment=True, attachment_filename=f'{stock_name}_stock_visualizations.png')
    return render_template("templates/index.html")

if __name__ == "__main__":
    app.run(debug=True)