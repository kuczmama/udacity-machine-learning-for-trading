import pandas as pd
import os
import matplotlib.pyplot as plt


def symbol_to_path(symbol, bas_dir='data'):
	"""Return CSV file path given ticker symbol"""
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
	"""Read stock data (adjusted close) for given symbols from CSV files. """
	df = pd.DataFrame(index=dates)
	if 'SPY' not in symbols: #Add SPY for reference if absent
		symbols.insert(0, 'SPY')

	for symbol in symbols:
		df.join(get_datatable(symbol), how='inner')
	return df

def get_max_close(symbol):
	return get_datatable(symbol)["Close"].max()

def get_datatable(symbol):
	return pd.read_csv("data/{}.csv".format(symbol))

def get_mean_volume(symbol):
	return get_datatable(symbol)["Volume"].mean()

def test_run():
	# df = pd.read_csv("data/SPY.csv")
	# df[['Close']].plot()
	# plt.show()
	    # Define a date range
    dates = pd.date_range('2010-01-22', '2010-01-26')

    # Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD']
    
    # Get stock data
    df = get_data(symbols, dates)
    print(df)

if __name__ == "__main__": 
	test_run()