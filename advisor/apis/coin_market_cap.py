import os
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from dotenv import load_dotenv
load_dotenv()

LIMIT = 5000

CMC_KEY = os.getenv("CMC_KEY")

def get_all():
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
    'start':'1',
    'limit': str(LIMIT),
    'convert':'USD'
    }
    headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': CMC_KEY,
    }

    session = Session()
    session.headers.update(headers)

    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)["data"]
        if response.status_code == 200:
            output = []
            for item in data:
                name = item["name"]
                print(name)
                output.append(name)
            with open("data/all_crypto.json", 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
            return output
    except (ConnectionError, Timeout, TooManyRedirects) as e:
        print(e)

    # def write_names_to_file(data:dict):


if __name__ == "__main__":
    get_all()