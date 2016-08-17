''' Analyze an entire portfolio '''
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
	allocs = pd.Series({'SPY':0.4,'XOM':0.4,'GLD':0.2})
	start_val = 1000000 # one million dollars!!!

	# Get data
	port_vals = util.portfolio_daily_values(start_val, dates, symbols, allocs)
	daily_returns = util.daily_returns(port_vals)
	print('Average Daily Returns:=',daily_returns.mean())
	print('Standard Daily Return:=',daily_returns.std())
	print('Sharpe Ratio:=', util.sharpe_ratio(daily_returns))
	daily_returns.plot()

	plt.show()

if __name__ == "__main__":
	test_run()