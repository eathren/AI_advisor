from fetch_stock_data import StockData, fetch_fresh_data, fetch_all_names
# from neural_net import NeuralNet

if __name__ == "__main__":
    fetch_fresh_data() # this makes an api call to every NASDAQ stock and updates to latest compact data
