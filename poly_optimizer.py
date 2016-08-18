''' Fit a line to a given set of data points using optimization '''
import pandas as pd 
import os
import matplotlib.pyplot as plt
import numpy as np
import util
import scipy.optimize as spo

def error_poly(C, data):
	"""Compute error between given polynomial and observed data.

	Parameters
	----------
	C: numpy.poly1d object or equivalent array representing polynomial coefficients
	data: 2D array where each row is a point (x, y)

	Returns error as a single real value.
	"""
	# Metric: Sum of swared Y-axis difference

	'''numpy.polyval(p, x)[source]
		Evaluate a polynomial at specific values.

		If p is of length N, this function returns the value:

		p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
	'''
	return np.sum((data[:,1] - np.polyval(C, data[:, 0]))**2)

def fit_poly(data, error_func, degree=3):
	"""Fit a polynomial to given data, using supplied error function.

	Parameters
	-----------
	 data: 2D array where each row is a point (x, y)
	 error_func: function that computest the error beween a polynomial and observed data

	 Returns polynomial that optimizes the error function.
	 """

	 # Generate initial guess polynomial model (all coeffs = 1)
	 Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

	 # Plot initial guess (optional)
	 x = np.linspace(-5, 5, 21)
	 plt.plot(x, np.polyval(Cguess,x), 'm--', linewidth=2.0, label="Initial guess")

	 # Call optimizer to minimize error function
	 result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'display':True})
	 return np.poly1d(result.x) # convert optimal result into a poly1d

def test_run():
	


if __name__ == "__main__":
	test_run()