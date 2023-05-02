from flask import Flask
from flask import render_template, request
import sys
sys.path.append('/Users/maximkiesel/PycharmProjects/Stock_Market_Prediction/Data')
from process_data import load_stock_data, cleaning_stock_data, save_data
#from visualizations import generate_visualizations
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO
from flask_socketio import run_in_thread



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_names = request.form["stock_name"]
        stock_name_list = stock_names.split(", ")[:5]
        dfs = load_stock_data(stock_name_list)
        dfs_cleaned = cleaning_stock_data(dfs, runs=1)

        image_data = {}
        for name in dfs_cleaned:
            # define area to calculate rolling average
            window_size = 20

            plt.figure(figsize=(25, 15))
            # calculate the rolling average
            rolling_mean_high = dfs_cleaned[name].rolling(window_size).mean()
            rolling_mean_low = dfs_cleaned[name].rolling(window_size).mean()

            # plot rolling mean variants
            plt.plot(rolling_mean_high, label="Rolling Mean High")
            plt.plot(rolling_mean_low, label="Rolling Mean Low")
            plt.legend(loc='upper left')
            plt.ylabel('Stock Price Rolling Average')
            plt.grid()

            # save the plot
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            image_data[name] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return render_template("index.html", result=dfs_cleaned, image_data=image_data)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)



