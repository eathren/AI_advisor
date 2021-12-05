import json

from stock_data import StockData, fetch_fresh_data, fetch_all_names, calc_all_risers_and_fallers
from neural_net import NeuralNet
from datetime import date


"""
This file serves as an entry point to create price predictions for the stocks that are expected to rise or fall the most 
in the current day, based on previous day values. 
"""


def calculate_with_net(data, direction="risers") -> dict:
    """
    name: calculate_with_net

    This function takes all the data from either risers or fallers, and runs them through a neural net
    to find price predictions and price difference.

    :param data:
    :param riser:
    :return output, a dict of predictions:
    """
    output = {}


    path = f"data/stocks/{direction}/{direction}.json"



    with open(path, "r") as f:
        data = json.load(f)

    for stock in data:
        # This calculates the daily stock price and prediction
        # Each neural net calculation takes about 1-5 minutes.
        # so run this BEFORE the markets open by at least 12 hours.
        try:
            # run and train the neural net. see neural_net.py
            id = stock
            net = NeuralNet(id)
            net.train()

            # getters
            previous_price = round(net.get_previous_price(), 2)
            predicted_price = round(net.get_predicted_price(), 2)
            # update predictions dict with that stocks' data.
            # difference = round(((predicted_price - predicted_price) / (previous_price + predicted_price)/2) * 100, 2)
            output[str(id)] = {"previous": previous_price, "predicted": predicted_price}
            print(output)
        except:
            print(f"Something happened during the neural net calculation for {id}")

        with open(f"data/stocks/predictions/{direction}/{date_today}.json", "w+") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)

    return output


if __name__ == "__main__":
    today = date.today()
    date_today = today.strftime("%Y-%m-%d")

    # fetch_fresh_data()  # this makes an api call to every NASDAQ stock and updates to latest compact data
    # this is designed to run every morning, or after previous market close.

    # risers, fallers = calc_all_risers_and_fallers()  # iterates thru all stocks and finds the ones with oscillators on extreme ends.
    with open("data/stocks/risers/risers.json", "r") as f:
        risers = json.load(f)
    with open("data/stocks/fallers/fallers.json", "r") as f:
        fallers = json.load(f)

    # this is where we'll save predictions
    predictions = {}
    predictions.update(calculate_with_net(data=risers, direction='risers'))
    predictions.update(calculate_with_net(data=fallers, direction='fallers'))

    # write all these predictions to a file.
    with open(f"data/stocks/predictions/{date_today}.json", "w+") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    print("Success!")

"""
To fetch stock names from nasdaq into json

echo "[\"$(echo -n "$(echo -en "$(curl -s --compressed 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt' | tail -n+2 | head -n-1 | perl -pe 's/ //g' | tr '|' ' ' | awk '{printf $1" "} {print $4}')\n$(curl -s --compressed 'ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt' | tail -n+2 | head -n-1 | perl -pe 's/ //g' | tr '|' ' ' | awk '{printf $1" "} {print $7}')" | grep -v 'Y$' | awk '{print $1}' | grep -v '[^a-zA-Z]' | sort)" | perl -pe 's/\n/","/g')\"]"

Source: https://quant.stackexchange.com/questions/1640/where-to-download-list-of-all-common-stocks-traded-on-nyse-nasdaq-and-amex

or:
https://www.nasdaq.com/market-activity/stocks/screener
"""