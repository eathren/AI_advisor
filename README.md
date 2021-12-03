## Stock AI and ML Advisor

This system is experimental, and stock prices are highly volatile. Please invest cautiously. We are not financial advisors, and nothing contained in the project is financial advice. 

### Overview
 
This project is designed to make stock recommendations based off of indicators and a ML neural net trained on previous data for each stock.

It runs by way of daily API calls to Alpha Advantage, and then crunches the data for each stock to find the most likely stocks that are under and overvalued (referedd to fallers and risers in ths program)

Then, it takes those stocks and runs neural nets on the most extreme outliers to predict the price for the following day, outputting that information to a file with the indicators and previous and predicted price.

### Requirements

If you want to run the demo of this program with 50 stocks, you will have to either use the provided Alpha Advantage api key in an .env file, or supply your own paid API key. The demo key has a limit of 500 calls a day, and is far to small to handle the 3500+ stocks on the nasdaq. 

Fetching stocks is also a slow process, since the API is limited to ~75 calls a minute, so api calls are limited to 1 a second.

### How to run

Install requirements with pip or pycharm.

If using pip, run <code> pip install requirements.txt</code>. This will install all packages listed in requirements.txt

**If You update or add a new package/library, run <code> pip freeze > requirements.txt </code>** to save the requirements for
others!

Using something like pycharm, pycharm wants to install for you, so you can do that there.

Project entry point is demo.py for limited use, and calculate_stocks_daily.py for full use. 

### Goals:

Take stock market and crypto data, and parse it into the most and least promising.

### Algorithms:

Use NumPy and Pandas to quickly solve bulk data sets and calculate out things like MACD, StochRSI, and hopefully things
like neural-net like datasets for recommendations.

The indicators that are used are RSI, Stoch RSI, EMA and SMA of varying time frames (7d, 14d, and 21d). 

### Project Due Date:

December 14, 2021

### Contributors:

Nolan Braman

Zuocheng Wang
