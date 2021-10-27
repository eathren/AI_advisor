import os
from dotenv import load_dotenv

load_dotenv()

CMC_KEY = os.getenv('CMC_KEY')

if __name__ == '__main__':
    print('PyCharm')
