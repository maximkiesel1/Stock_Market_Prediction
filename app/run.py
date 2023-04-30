from flask import Flask
from flask import render_template, request
import sys
sys.path.append('/Users/maximkiesel/PycharmProjects/Stock_Market_Prediction/Data')
from process_data import load_stock_data, cleaning_stock_data, save_data
#from visualizations import generate_visualizations
import matplotlib.pyplot as plt
import os



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock_names = request.form["stock_name"]
        stock_name_list = stock_names.split(", ")[:5]
        file = request.form['database_filepath']
        dfs = load_stock_data(stock_name_list)
        dfs_cleaned = cleaning_stock_data(dfs, runs=1)
        save_data(dfs_cleaned, file)
        result = "Data saved successfully!"
        return render_template("index.html", result1=result)
    return render_template("index.html")


@app.route("/visualizations", methods=["GET", "POST"])
def visualizations():
    if request.method == "POST":
        data = request.form["generate_visu"]

        #file_names = generate_visualizations(data)

        file_names = []
        for stock_name in data:
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

            # save the plot
            filename = f'{stock_name}_stock.png'
            filepath = os.path.join(app.static_folder, filename)
            plt.savefig(filepath)
            file_names.append(filename)

        return render_template("index.html", result2=file_names)#'Visualization generated!')
    return render_template("index.html")


def my_function(stock_name):
    # hier können Sie den eingegebenen Text des Benutzers verwenden, um Ihr anderes Programm zu starten
    # und das Ergebnis zurückgeben
    return f"You entered {stock_name}"


if __name__ == "__main__":
    app.run(debug=True)