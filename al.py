import alpaca_trade_api as tradeapi
import requests, json, csv
import pandas as pd
# authentication and connection details
api_key = "AKRTT2V2ZQH4FG7LGHJC"
api_secret = "jyEuhsD7AjxhazPHgAnPA4lmnFayf1IeUjcPHHMw"
base_url = "https://api.alpaca.markets"

# instantiate REST API
api = tradeapi.REST(api_key,  api_secret, api_version="v2")
aapl = api.get_barset('AAPL', 'day')
# obtain account information
account = api.get_account()
print(aapl.df)
aapl.to_json("Test.json")
# print(account)