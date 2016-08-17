''' Build a dataframe in pandas '''
import pandas as pd 
import os
import matplotlib.pyplot as plt

import util

def test_run():
	start_date = '2007-01-01'
	end_date = 	'2013-08-31'
	dates = pd.date_range(start_date, end_date)
	symbols = ['SPY']

	# Get data
	df = util.get_data(symbols, dates)
	util.plot_data(df, "SPY")

	# plot daily returns
	daily_returns = util.compute_daily_returns(df)
	util.plot_data(daily_returns, "Daily Returns")

	# Plot histogram
	daily_returns.hist(bins=20)

	# Get Mean and Std Deviation
	mean = daily_returns['SPY'].mean()
	print("mean=", mean)
	std = daily_returns['SPY'].std()

	plt.axvline(mean, color='w', linestyle='dashed', linewidth=2)
	plt.axvline(std, color='y', linestyle='dashed', linewidth=2)
	plt.axvline(-std, color='y', linestyle='dashed', linewidth=2)
	plt.axvline(2*std, color='y', linestyle='dashed', linewidth=2)
	plt.axvline(-2*std, color='y', linestyle='dashed', linewidth=2)

	# Compute curtosis
	kurtosis = daily_returns.kurtosis()
	print("kurtosis=", kurtosis)

	plt.show()


if __name__ == "__main__":
	test_run()