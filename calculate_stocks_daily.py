import json

from fetch_stock_data import StockData, fetch_fresh_data, fetch_all_names, calc_all_risers_and_fallers
from datetime import date, timedelta

# from neural_net import NeuralNet

"""
This file serves as an entry point to create price predictions for the stocks that are expected to rise or fall the most 
in the current day, based on previous day values. 

"""

if __name__ == "__main__":
    date_today = today.strftime("%Y-%m-%d")

    fetch_fresh_data()  # this makes an api call to every NASDAQ stock and updates to latest compact data
    # this is designed to run every morning, or after previous market close.

    risers, fallers = calc_all_risers_and_fallers()

    # this is where we'll save predictions
    predictions = {}

    for stock in risers:
        # This calculates the daily stock price and prediction
        # Each neural net calculation takes about 5 minutes on my laptop.
        # so run this BEFORE the markets open by at least 12 hours.
        try:
            id, previous_price, predicted_price = NeuralNet(stock, full=True, new_data=False)
            predictions[id] = {'previous': previous_price, 'predicted': predicted_price}
        except:
            print(f"Something happened during the neural net calculation for {id}")
    # write all these predictions to a file.

    with open(f"data/stocks/predictions/{date_today}.json", "w+") as f:
        with open('data/stocks/risers/risers.json', 'w', encoding='utf-8') as f:

    print("Success!")
