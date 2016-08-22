''' Fit a line to a given set of data points using optimization '''
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
import util
import scipy.optimize as spo

def fit_line(data, func, start_val, dates, symbols, allocs):
	"""cipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)[source]
		Minimization of scalar function of one or more variables.

		In general, the optimization problems are of the form:

		minimize f(x) subject to

		g_i(x) >= 0,  i = 1,...,m
		h_j(x)  = 0,  j = 1,...,p
		where x is a vector of one or more variables. g_i(x) are the inequality constraints. h_j(x) are the equality constrains.

	Optionally, the lower and upper bounds for each element in x can also be specified using the bounds argument.
	"""
	bounds = []
	for x in range(0, len(symbols)):
		bounds.append((0.0, 1.0))
	print(bounds)
	allocs = spo.minimize(negative_sharpe, allocs, 
		args=(data, start_val, dates, symbols),
		bounds=bounds,
		constraints={'type': 'eq', 'fun': lambda x:  np.sum(x) - 1},
		method='SLSQP')
	# print('result=', allocs.x)
	return allocs.x

def negative_sharpe(allocs, data, start_val, dates, symbols):
	"""
	C: numpy.poly1d object or equivalent array representing polynomial coefficients
	data: 2D array where each row is a point (x, y)

	Returns error as a single real value.
	"""
	result = util.sharpe_ratio(util.portfolio_daily_values(start_val, dates, symbols, allocs))
	#print('result: ', result)
	return result
	#return util.portfolio_daily_values(start_val, dates, symbols, allocs).mean()

def test_run():
	# Get initial data
	start_date = '2004-08-31'
	end_date = 	'2016-01-01'
	dates = pd.date_range(start_date, end_date)
	symbols = ['SPY','VOO','GLD','XOM','AAPL']
	original_allocation =  [0.2, 0.2, 0.2, 0.2, 0.2]
	start_val = 20000 # one million dollars!!!

	# plot the original data
	original_data = util.portfolio_daily_values(start_val, dates, symbols, original_allocation)

	util.plot_data(original_data, "Original", label='Original')


	# Plot the new data
	new_allocation = fit_line(original_data, negative_sharpe, start_val, dates, symbols, original_allocation)
	new_data = util.portfolio_daily_values(start_val, dates, symbols, new_allocation)
	util.plot_data(new_data, "Optimized", label='Optimized')
	plt.legend(loc='upper right')
	
	print("Original Allocation=", original_allocation)
	print("New allocation=", new_allocation)

	plt.show()



if __name__ == "__main__":
	test_run()