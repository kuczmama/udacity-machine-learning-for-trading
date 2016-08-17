''' Build a dataframe in pandas '''
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np

import util

def test_run():
	start_date = '2010-08-31'
	end_date = 	'2013-01-01'
	dates = pd.date_range(start_date, end_date)
	symbols = ['SPY','XOM', 'GLD']

	# Get data
	df = util.get_data(symbols, dates)

	# Compute daily returns
	daily_returns = util.compute_daily_returns(df)
	daily_returns.plot(kind='scatter', x='SPY', y='XOM')
	daily_returns['SPY'].hist()
	daily_returns['XOM'].hist()
	beta_XOM, alpha_XOM = np.polyfit(daily_returns['SPY'], daily_returns['XOM'], 1)
	# plt.plot(daily_returns['SPY'], beta_XOM*daily_returns['SPY'] + alpha_XOM, '-', color='r')
	# plt.show()

	# SPY v GOLD scatterplot
	#daily_returns.plot(kind='scatter', x='SPY', y='GLD')
	plt.show()

if __name__ == "__main__":
	test_run()