import csv
import json
import os

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

AA_KEY = os.getenv('AA_KEY')


def get_response(name):
    """
        name: get_response

        This function fetches all the stock data for a given stock ID.

        :return pandas dataframe timeseries stock data:
        """
    params = {'symbol': name, 'apikey': AA_KEY, 'outputsize': 'full'}
    response = requests.get(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED",
        params=params)
    print(response.url)
    if response.status_code == 200:
        # turn json to pd dataframe that is renamed
        return json_to_pd_df(response.json())
    else:
        print(f"An error has occured with fetching the json data for {name}")


def json_to_pd_df(json):
    """
    name: json_to_pd_df
    Turns the json data into a pandas datafram
    :param self.json, json stock data from Alpha Advantage.
    :return df, a PD dataframe object:
    """
    # Pandas is dumb when it comes to renaming rows. Make them columns to rename instead.
    df = pd.DataFrame(json['Time Series (Daily)']).transpose()
    # rename columns, since the api has a terrible naming convention.
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                            '5. adjusted close': 'adjusted close', '6. volume': 'volume',
                            '7. dividend amount': 'dividend amount', '8. split coefficient': 'split coefficient'})
    # change datatypes from Objects to floats
    return df.astype(float).iloc[::-1]


if __name__ == "__main__":
    get_response("INTU")
