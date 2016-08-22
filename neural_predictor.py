import os
import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd 

learning_rate = 0.1
start_date = '2010-08-31'
end_date = 	'2013-01-01'
dates = pd.date_range(start_date, end_date)

"""How data is stored
	
	weights are lined up with inputs

	Inputs: 
	Dates input1 input2 input3 ... input N
	1/1		 3.0    2.0  1.5       1.7
	1/2      2.5    1.8  1.3       2.0

	Weights:
	[weight1, weight2, weight3... weight N]

	Expected:
	[expected1, expected2, expected3 ... expected N]
"""

def create_weights(inputs):
	return pd.Series([0.0] * inputs.shape[1])

def update_weight(weight, delta):
	weight += delta

def delta_weight(date, column, inputs, expected, weights):
	# delta_weight = learning_rate(expected - actual) input_i
	expected = expected.ix[date][0]
	return learning_rate * (expected - output(date, inputs, weights)) * inputs.ix[date][column]

def output(index, inputs, weights):
	# sum(weights * inputs)
	output = (inputs.ix[index] * weights).sum()
	return output


def train(inputs, weights, expected):
	d_weight = 0.0
	#iterate through each row
	for index, row in inputs.iterrows():
		# Iterate throught the columns
		for column in range(row.shape[0]):
			d_weight = delta_weight(index,column,inputs, expected, weights)
			weights.ix[column] = (weights.ix[column] + d_weight)

def test_run():
	num_inputs = 4
	inputs = pd.DataFrame(index=dates)

	# Create random inputs
	for x in range (num_inputs):
		tmp = pd.DataFrame(data=np.random.rand(len(dates)), index=dates)
		tmp = tmp.rename(columns={0: x})
		inputs = inputs.join(tmp)
	expected = pd.DataFrame(data=np.random.rand(len(dates)), index=dates)

	weights = create_weights(inputs) # start all weights at 0
	train(inputs, weights, expected)
	print(weights)




if __name__ == "__main__":
	test_run()