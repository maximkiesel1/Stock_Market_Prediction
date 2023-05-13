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
    - keras_model
    - `modelling.py`
    - visual_predict
    - visual_validation

You can find the technical analyses for the code in the Jupyter notebooks. 

# How to use the program

To use the package, follow these steps:

- Clone the repository to your local machine.

Please ensure that the libraries from the requirements.txt file are installed on your system before running the code. If any of these libraries are missing, you can install them using pip. For example, to install the pandas library, you can use the following command:

```
pip install pandas
```

- GET THE DATA: Navigate to the `data` folder and run `python process_data.py`. This will start a input box to write the desired stock names (acronym) and  get the historical data automaticlly from the yahoo API, clean the data, and save the resulting data in a SQL database called `cleaned_data_sql.db`.
  - Here is an example to run the program:

<img width="810" alt="Bildschirmfoto 2023-05-13 um 13 51 27" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/838eaccf-192f-4a3c-b707-356e643a3d88">
<br>
<img width="486" alt="Bildschirmfoto 2023-05-13 um 13 51 55" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/989f285b-e125-43cc-993b-723fbce8e3b0">
<br>
<img width="409" alt="Bildschirmfoto 2023-05-13 um 13 52 17" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/5ac89906-e6f8-4194-b5e2-afbed276cc6a">
<br>
<img width="539" alt="Bildschirmfoto 2023-05-13 um 13 53 56" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/054e12ce-6084-418d-a51b-b84c5ea858a9">
<br>
<img width="600" alt="Bildschirmfoto 2023-05-13 um 13 54 05" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/854d00c9-2911-4fa3-a9f3-5141ef38fb29">
<br>
<img width="568" alt="Bildschirmfoto 2023-05-13 um 13 52 51" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/2d845d58-8231-44a2-9aa8-764087a93282">
<br>

- DO THE PREDICTION: Navigate to the `model` folder and run `python modeling.py`.
  - Here is an example to run the program:
 
<img width="843" alt="Bildschirmfoto 2023-05-13 um 13 53 15" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/af7ca988-4d6c-40e0-8181-2bf1c14f3472">
<br>
<img width="458" alt="Bildschirmfoto 2023-05-13 um 13 53 31" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/21d6b55b-fd46-4e39-8d58-fe7fbb45fb3f">
<br>
<img width="467" alt="Bildschirmfoto 2023-05-13 um 13 56 51" src="https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/8a9e8f55-0033-481c-814e-c7c660f342d1">
<br>

# Findings
Growth stocks and the small-cap stocks are worse than blue-chip and dividend stocks in their performance.

XXXXXXXX PCITURES XXXXXXX

# Further Informations
- Next Step: Create a WebApp with Flask for the user
- Bugs: 
   - It is not possible to create an pickle file for the ML model
   - While training the model, this error messages appears (without an impact):
      - `WARNING:absl:Found untraced functions such as lstm_cell_68_layer_call_fn, lstm_cell_68_layer_call_and_return_conditional_losses while saving (showing 2 of 2). These functions will not be directly callable after loading.`
   - For stock abbreviations with a hyphen like "BCT-USD" and a sql data bank with my code has already been created once, then the database must be deleted manually, otherwise this stock entry cannot be deleted via my code. All other shares will be deleted automatically when restarting the code.


# License
This project is licensed under the MIT license.

The MIT License is a permissive open source license that allows for unlimited distribution and reuse of the licensed software. It is short and easy to understand, and it places very few restrictions on what you can do with the software. You can read the full text of the MIT License at https://opensource.org/licenses/MIT.

# **Risk Disclaimer**:

Trading in the financial markets involves a high degree of risk and may not be suitable for all investors. The trading machine learning algorithm presented herein is an experimental program and is provided on an "as is" basis without any warranties, expressed or implied. The creators and owners of the algorithm do not make any representations or warranties, either express or implied, as to the accuracy, reliability, completeness, or appropriateness for any particular purpose of the information, analyses, algorithms, or models contained in this algorithm.

The creators and owners of the algorithm do not accept any liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information.

Trading and investment decisions are the sole responsibility of the user. It is the responsibility of the user to perform proper due diligence before making any investment or trading decisions. Users should seek professional advice before trading in the financial markets.

Past performance of the algorithm is not indicative of future results. The user assumes full responsibility and risk of using this algorithm. By using this algorithm, the user agrees to these terms and conditions and accepts full responsibility for all trading and investment decisions made using this algorithm.
