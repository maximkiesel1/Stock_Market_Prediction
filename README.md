# Stock Price Prediction

![nick-chong-N__BnvQ_w18-unsplash(1)](https://github.com/maximkiesel1/Stock_Market_Prediction/assets/119667336/5b5f83b2-f126-4cc1-9ba9-a048d2931885)

# Stock Price Prediction With LSTM

In this repository, I will create a machine learning algorithm to predict the stock price (Adjusted Close) and make a trading recommendation for the user.

The LSTM (Long Short-Term Memory) algorithm will be used. This is a popular algorithms for stock prediction.

The goal is to create an algorithm which can predict the stock prices with a tolerance of ± 5%. Also, I want to show how the different stock categories influence the accuracy of the algorithm.

The following stock categories will be analyzed:
- Blue chip stocks: Blue chip stocks are shares of large, established companies with stable financial performance and low risk.
   - BMW (BMW.DE)
- Growth stocks: growth stocks are stocks of companies with high potential for future growth. These stocks often carry higher risks, but also offer higher potential returns
    - Tesla, Inc. (TSLA)
    - Bitcoin (BTC-USD)
- Dividend stocks: dividend stocks are stocks of companies that pay regular dividends to their shareholders. These stocks often carry lower risks and offer a regular source of income.
    - Johnson & Johnson (JNJ)
- Small-cap stocks: small-cap stocks are stocks of small companies with higher risk and higher potential for growth and return.
    - Etsy, Inc (ETSY)

# Technical Concepts

- Why does it exist?
- Frame your project for the potential user. 
- Compare/contrast your project with other, similar projects so the user knows how it is different from those projects.
- Highlight the technical concepts that your project demonstrates or supports. Keep it very brief.
- Keep it useful.

# Getting Started
Include any essential instructions for:
- Getting it
- Installing It
- Configuring It
- Running it

# More Specific Topics (+ sample sub-categories)
- Versioning: Services, APIs, Systems
- Common Error Messages/related details
- Tests

# Findings
Growth stocks and the small-cap stocks are worse than blue-chip and dividend stocks in their performance.


# Further Informations
- Next Step: Create a WebApp with Flask for the user
- Bugs: 
   - it is not possible to create an pickle file for the ML model

# License
This project is licensed under the MIT license.

The MIT License is a permissive open source license that allows for unlimited distribution and reuse of the licensed software. It is short and easy to understand, and it places very few restrictions on what you can do with the software. You can read the full text of the MIT License at https://opensource.org/licenses/MIT.

# **Risk Disclaimer**:

Trading in the financial markets involves a high degree of risk and may not be suitable for all investors. The trading machine learning algorithm presented herein is an experimental program and is provided on an "as is" basis without any warranties, expressed or implied. The creators and owners of the algorithm do not make any representations or warranties, either express or implied, as to the accuracy, reliability, completeness, or appropriateness for any particular purpose of the information, analyses, algorithms, or models contained in this algorithm.

The creators and owners of the algorithm do not accept any liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information.

Trading and investment decisions are the sole responsibility of the user. It is the responsibility of the user to perform proper due diligence before making any investment or trading decisions. Users should seek professional advice before trading in the financial markets.

Past performance of the algorithm is not indicative of future results. The user assumes full responsibility and risk of using this algorithm. By using this algorithm, the user agrees to these terms and conditions and accepts full responsibility for all trading and investment decisions made using this algorithm.
