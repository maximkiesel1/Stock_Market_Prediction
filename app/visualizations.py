import matplotlib.pyplot as plt
import os


def generate_visualizations(data):

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

    return file_names
