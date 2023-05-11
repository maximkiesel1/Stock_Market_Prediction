# Stock Price Prediction

![nick-chong-N__BnvQ_w18-unsplash(1)](https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/5b5f83b2-f126-4cc1-9ba9-a048d2931885)

# Introduction

In this repository, I will create a machine learning algorithm to predict the stock price (Adjusted Close) and make a trading recommendation for the user.

The LSTM (Long Short-Term Memory) algorithm will be used. This is a popular algorithm for stock prediction.

The goal is to create an algorithm that can predict stock prices with a tolerance of ± 5%. I also want to show how the different stock categories affect the accuracy of the algorithm.

The following stock categories will be analyzed
- Blue chip stocks: Blue chip stocks are stocks of large, established companies with stable financial performance and low risk.
   - BMW (BMW.DE)
   
- Growth stocks: Growth stocks are stocks of companies with high potential for future growth. These stocks often carry higher risks, but also offer higher potential returns.
    - Tesla, Inc. (TSLA)
    - Bitcoin (BTC-USD)
   
- Dividend stocks: Dividend stocks are stocks of companies that pay regular dividends to their shareholders. These stocks are often less risky and provide a regular source of income.
    - Johnson & Johnson (JNJ)
    
- Small-cap stocks: Small-cap stocks are stocks of small companies with higher risk and higher potential for growth and return.
    - Etsy, Inc (ETSY)

# Technical Concepts

This project involves the following technical concepts:

- Used API: The Yahoo Finance API was used to collect historical stock prices data.
   - Using always the last three years to the present date for the training.

- Predicting Stock Prices: The aim of the project is to predict the Adjusted Close stock price for 7, 14, and 28 days in the future.
   - Allowing 3 Stocks for prediction, because of the long training duration
   - 20 randowm iteration with different hyperparameter: 
      - Hyperparameters: `params = {
        'window_size': [4, 8, 12, 20],
        'lstm_units': [16, 32, 64, 128],
        'dense_units': [16, 32, 64, 128],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1]}`
   - Training 200 epochs 

- Accuracy: The model is designed to achieve an accuracy of approximately +-5%.

- Recurrent Neural Networks (RNN) LSTM: In order to achieve the prediction task, the project uses Recurrent Neural Networks (RNN) in form of the LSTM algorithm with the Keras deep learning library. LSTM is particularly suited for processing sequential data, making them a good fit for predicting stock prices. The Keras library provides an easy-to-use interface for building and training deep learning models.

# Directory Structure

The package has the following directory structure:

 - data
   - `process_data.py`
   - `cleaned_data_sql.db`
  - models
    - `train_classifier.py`
    - `text_length_extractor.py`
    - `classifier.pkl`


# How to use the program

To use the package, follow these steps:

- Clone the repository to your local machine.

Please ensure that the libraries from the requirements.txt file are installed on your system before running the code. If any of these libraries are missing, you can install them using pip. For example, to install the pandas library, you can use the following command:

```
pip install pandas
```

- Navigate to the `data` folder and run `python process_data.py`. This will start a input box to write the desired stock names (acronym) and  get the historical data automaticlly from the yahoo API, clean the data, and save the resulting data in a SQLite database called `cleaned_data_sql.db`.
  - Here is an example to run the program:

# More Specific Topics (+ sample sub-categories)
- Versioning: Services, APIs, Systems
- Common Error Messages/related details
- Tests

# Findings
Growth stocks and the small-cap stocks are worse than blue-chip and dividend stocks in their performance.


# Further Informations
- Next Step: Create a WebApp with Flask for the user
- Bugs: 
   - It is not possible to create an pickle file for the ML model
   - While training the model, this error messages appears (without an impact):
      - `WARNING:absl:Found untraced functions such as lstm_cell_68_layer_call_fn, lstm_cell_68_layer_call_and_return_conditional_losses while saving (showing 2 of 2). These functions will not be directly callable after loading.`


# License
This project is licensed under the MIT license.

The MIT License is a permissive open source license that allows for unlimited distribution and reuse of the licensed software. It is short and easy to understand, and it places very few restrictions on what you can do with the software. You can read the full text of the MIT License at https://opensource.org/licenses/MIT.

# **Risk Disclaimer**:

Trading in the financial markets involves a high degree of risk and may not be suitable for all investors. The trading machine learning algorithm presented herein is an experimental program and is provided on an "as is" basis without any warranties, expressed or implied. The creators and owners of the algorithm do not make any representations or warranties, either express or implied, as to the accuracy, reliability, completeness, or appropriateness for any particular purpose of the information, analyses, algorithms, or models contained in this algorithm.

The creators and owners of the algorithm do not accept any liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information.

Trading and investment decisions are the sole responsibility of the user. It is the responsibility of the user to perform proper due diligence before making any investment or trading decisions. Users should seek professional advice before trading in the financial markets.

Past performance of the algorithm is not indicative of future results. The user assumes full responsibility and risk of using this algorithm. By using this algorithm, the user agrees to these terms and conditions and accepts full responsibility for all trading and investment decisions made using this algorithm.
