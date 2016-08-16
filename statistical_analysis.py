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


def get_rolling_std(df, window=20):
	return pd.Series.rolling(df, center=False, window=window).std()

def get_rolling_mean(df, window=20):
	return pd.Series.rolling(df, center=False,window=window).mean()

def get_bollinger_bands(rm, rstd):
	upper_band = rm + (rstd * 2)
	lower_band = rm - (rstd * 2)
	return upper_band, lower_band

def test_run():
	start_date = '2016-01-01'
	end_date = 	'2018-08-31'
	dates = pd.date_range(start_date, end_date)
	symbols = ['GOOG', 'IBM', 'GLD']

	# Read in more stocks
	df = get_data(symbols, dates)
	# print(df.ix[start_date:end_date, ['GOOG','IBM']])
	# Compute global statistics for each stock
	#print("\nMean:\n", df.mean())
	#print("\nMedian:\n ", df.median())
	#print("\nStd Dev:\n", df.std())

	# plot_data(df,'Stock Prices','Dates','Price',False)
	ax = df['SPY'].plot(title='SPY', label='SPY')

	# Plot rolling mean
	rm = get_rolling_mean(df['SPY'])
	rstd = get_rolling_std(df['SPY'])
	upper_band, lower_band = get_bollinger_bands(rm, rstd)

	upper_band = upper_band.rename(columns={'SPY':'SPY Upper STD'})
	lower_band = lower_band.rename(columns={'SPY':'SPY Lower STD'})



	# Add rolling mean to same plot
	rm.plot(label='Rolling Mean ', ax=ax)
	upper_band.plot(label='Upper Band', ax=ax)
	lower_band.plot(label='Lower Band', ax=ax)
	ax.legend(loc='upper left')
	plt.show()


if __name__ == "__main__":
	test_run()