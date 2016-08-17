''' Build a dataframe in pandas '''
import pandas as pd 
import os
import matplotlib.pyplot as plt

import util

def test_run():
	start_date = '2007-01-01'
	end_date = 	'2013-08-31'
	dates = pd.date_range(start_date, end_date)
	symbols = ['SPY','XOM']

	# Get data
	df = util.get_data(symbols, dates)
	# util.plot_data(df, "SPY")

	# Compute daily returns
	daily_returns = util.compute_daily_returns(df)
	# util.plot_data(daily_returns, title='Daily Returns', ylabel='Daily Returns')

	daily_returns['SPY'].hist(bins=20, label='SPY')
	daily_returns['XOM'].hist(bins=20, label='XOM')
	plt.legend(loc='upper right')

	plt.show()


if __name__ == "__main__":
	test_run()