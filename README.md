## Stock and Crypto Advisor

### How to run

If you don't have pip installed, install pip (although it should have been added when you installed python)

Install requirements. You should consider doing it in something like an virtual env so you don't clutter your pc.

https://docs.python.org/3/tutorial/venv.html

Next, run <code> pip install requirements.txt</code>. This will install all packages listed in requirements.txt

**If You update or use a new package, run <code> pip freeze > requirements.txt </code>** to save the requirements for
others!

Using something like pycharm, pycharm wants to install for you, so you can do that there.

### Project structure

right now JSON data is saved in the data folder. This is right now used to hold some crypto data to avoid abusing the
coingecko api, and in the future will hold stocks that the user holds (perhaps), to make recommendations of when to
sell.

This folder exists since it is so easy to write to files in python, and setting up an actual DB can be a huge pain.

### Goals:

Take stock market and crypto data, and parse it into the most and least promising.

### Algorithms:

Use NumPy and Pandas to quickly solve bulk data sets and calculate out things like MACD, StochRSI, and hopefully things
like neural-net like datasets for recommendations.

### Project Due Date:

December 14, 2021

### Contributors:

Nolan Braman

