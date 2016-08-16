''' Build a dataframe in pandas '''
import pandas as pd 
import os
import matplotlib.pyplot as plt

def symbol_to_path(symbol, base_dir="data"):
	""" Return CSV file path given ticker symbol"""
	return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def normalize_data(df):
	return df / df.ix[0, :]


def plot_data(df, title="Stock Prices", xlabel="Dates", ylabel="Price", normalize=False):
	'''Plot Stock Prices'''
	if normalize:
		df = normalize_data(df)

	ax = df.plot(title=title, fontsize=20)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	plt.show()

def plot_selected(df, columns, start_index, end_index):
	plot_data(df[start_index:end_index, columns])

def get_data(symbols, dates):
	df = pd.DataFrame(index=dates)
	if 'SPY' not in symbols:
		symbols.insert(0, 'SPY')

	for symbol in symbols:
		df_temp = pd.read_csv(
			symbol_to_path(symbol),
			index_col="Date",
			parse_dates=True,
			usecols=["Date", "Adj Close"],
			na_values=['nan'],
			engine='python'
		)
		df_temp = df_temp.rename(columns={'Adj Close': symbol})
		df = df.join(df_temp)

	df = df.dropna(subset=["SPY"])
	return df

def test_run():
	start_date = '2010-01-01'
	end_date = 	'2010-12-31'
	dates = pd.date_range(start_date, end_date)
	symbols = ['GOOG', 'IBM', 'GLD']

	# Read in more stocks
	df = get_data(symbols, dates)
	# print(df.ix[start_date:end_date, ['GOOG','IBM']])
	plot_data(df,'Stock Prices','Dates','Price',False)

if __name__ == "__main__":
	test_run()