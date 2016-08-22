import os
import matplotlib.pyplot as plt
import numpy as np
import util
import pandas as pd 

learning_rate = 0.1
start_date = '2010-08-31'
end_date = 	'2013-01-01'

"""How data is stored
	
	weights are lined up with inputs

	Inputs: 
	Dates input1 input2 input3 ... input N
	1/1		 3.0    2.0  1.5       1.7
	1/2      2.5    1.8  1.3       2.0

	Weights:
	[weight1, weight2, weight3... weight N]
"""

def create(dates, inputs):
	inputs = pd.DataFrame(index=dates)
	weights = pd.Series([0.0] * inputs.shape.x)

def update_weight(weight, delta):
	weight += delta

def update_weights():
	# delta_weight = learning_rate(expected - actual) input_i
	pass

def output():
	# for
	pass

def test_run():
	dates = pd.date_range(start_date, end_date)
	inputs = pd.DataFrame(index=dates)
	num_inputs = 10

	for x in range (num_inputs):
		tmp = pd.DataFrame(data=np.random.rand(len(dates)), index=dates)
		tmp = tmp.rename(columns={0: x})
		inputs = inputs.join(tmp)

	

	print(inputs)



if __name__ == "__main__":
	test_run()