import csv
import json
import os

import requests
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

from dotenv import load_dotenv

load_dotenv()
AP_KEY = os.getenv('APCA_API_KEY_ID')
AP_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('APCA_API_BASE_URL')

api = REST(key_id=AP_KEY, secret_key=AP_SECRET,
           base_url=BASE_URL, api_version="v2")


def get_response(name, time_scale='1D', limit=1000):
    return api.get_barset(name, time_scale, limit=limit).df[name]


if __name__ == "__main__":
    print(get_response("AAPL"))
