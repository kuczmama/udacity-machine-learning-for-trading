''' Fit a line to a given set of data points using optimization '''
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
import util
import scipy.optimize as spo

def fit_line(data, error_func):
	"""Fit a line to the given data given the supplied error function
	Parameters
	----------
	data: 2D array where each row is a point (X0, Y)
	error_func: function that computes the error between a line and the observed data

	Returns the line that minimizes the error function
	"""

	# Generate initial guess for the model
	l = np.float32([0, np.mean(data[:, 1])]) # slope = 0, intercept = mean(y values)

	# plot initial guess (optional)
	x_ends = np.float32([-5, 5])
	plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

	# Call optimizer to minimize error function
	result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'display':True})
	return result.x


def error(line, data):
	"""Compute error between given line model and observed data.

	Parameters
	-----------
	line: tuple/list/array (C0, C1) where CO is slope and C1 is Y-intercept
	data: 2D array where each row is a point (x, y)

	Returns error as a single real value

	error = (y - (mx+ b)) ^2 (The Y-squared difference from the line)
	y = data[:, 1]
	x = data[:, 0]
	m = line[0]
	b = line[1]

	"""

	# Metric: Sum of squared Y-Axis difference
	return np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1]))**2)

def test_run():
	# define the original line
	l_orig = np.float32([4, 2]) # Where y = mx + b <==> slope: y = 4x + 2
	print("original line: m or c0={}, b or c1={}".format(l_orig[0], l_orig[1]))

	"""
	numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
	Return evenly spaced numbers over a specified interval.

	Returns num evenly spaced samples, calculated over the interval [start, stop].

	The endpoint of the interval can optionally be excluded.
	"""

	x_orig = np.linspace(0, 10, 21) # Returns an ndarray from 0-10 with 21 numbers evenly spaced
	y_orig = l_orig[0] * x_orig + l_orig[1] # original y value from line
	plt.plot(x_orig, y_orig, 'b--', linewidth=2.0, label='Original Line') #plot the original line... need a bunch of points to plot


	# Generate noisy data points
	"""
	numpy.random.normal(loc=0.0, scale=1.0, size=None)
	Draw random samples from a normal (Gaussian) distribution.

	The probability density function of the normal distribution, first derived by De Moivre and 200 years later by both Gauss and Laplace independently [R250], is often called the bell curve because of its characteristic shape (see the example below).

	The normal distributions occurs often in nature. For example, it describes the commonly occurring distribution of samples influenced by a large number of tiny, random disturbances, each with its own unique distribution [R250].

	Parameters:	
	loc : float
	Mean (“centre”) of the distribution.
	scale : float
	Standard deviation (spread or “width”) of the distribution.
	size : int or tuple of ints, optional
	Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
	"""
	noise_sigma = 3.0
	noise = np.random.normal(0, noise_sigma, y_orig.shape)
	"""
	numpy.asarray(a, dtype=None, order=None)[source]
	Convert the input to an array.

	Parameters:	
	a : array_like
	Input data, in any form that can be converted to an array. This includes lists, lists of tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
	dtype : data-type, optional
	By default, the data-type is inferred from the input data.
	order : {‘C’, ‘F’}, optional
	Whether to use row-major (C-style) or column-major (Fortran-style) memory representation. Defaults to ‘C’.
	Returns:	
	out : ndarray
	Array interpretation of a. No copy is performed if the input is already an ndarray. If a is a subclass of ndarray, a base class ndarray is returned.
	"""
	data = np.asarray([x_orig, y_orig + noise]).T # Transpose the data to make it fit
	plt.plot(data[:,0], data[:,1], 'go', label='Data points')

	# Try to fit a line to this data
	l_fit = fit_line(data, error)

	print("Fitted Line: m = {}, b = {}".format(l_fit[0], l_fit[1]))

	plt.plot(x_orig, l_fit[0]*x_orig + l_fit[1], 'b--')
	plt.show()

if __name__ == "__main__":
	test_run()