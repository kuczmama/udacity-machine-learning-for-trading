import os
import numpy as np
import util
import pandas as pd 

learning_rate = 0.01


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
	#print(expected.ix[date])
	# delta_weight = learning_rate(expected - actual) input_i
	expected = expected.ix[date][0]
	#@print(expected[date])
	return learning_rate * (expected - output(date, inputs, weights)) * inputs.ix[date][column]
	#return 0

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

# Return expected values in a datatable 1 if market went up.  0 if market went down
def get_expected(dates):
	expected = util.get_data(['SPY'], dates)
	expected = util.daily_returns(expected)
	expected = expected.applymap(lambda x: 1.0 if x > 0 else 0.0)
	expected = expected.rename(columns={'SPY': 'Expected'})
	#print(expected)
	return expected
	
def get_rolling_over_std(df):
		# number of std deviations away from rolling mean
	rolling_over_std = util.get_rolling_mean(df)/util.get_rolling_std(df) 
	rolling_over_std = rolling_over_std.shift(1)
	rolling_over_std = rolling_over_std.rename(columns={'SPY': 'Num STDS'})
	return rolling_over_std
	
def test_results(weights):
	start_date = '2013-01-02'
	end_date = '2016-01-01'
	dates = pd.date_range(start_date, end_date)
	spy = util.get_data(['SPY'], dates)
	normed_spy = util.normalize_data(spy)
	percent_correct = 0.0
	for date, row in normed_spy.iterrows():
		print(date, row)
	
	pass

def test_run():
	start_date = '1993-08-31'
	end_date = 	'2013-01-01'
	dates = pd.date_range(start_date, end_date)
	num_training_inputs = 1
	
	expected = get_expected(dates) # From SPY
	
	training_inputs = pd.DataFrame(index=dates)
	spy = util.get_data(['SPY'], dates)
	normed_spy = util.normalize_data(spy)
	training_inputs = training_inputs.join(spy)
	# Create random training_inputs
	
	rolling_over_std = get_rolling_over_std(normed_spy)
	
	for x in range (num_training_inputs):
		tmp = pd.DataFrame(data=np.random.rand(len(dates)), index=dates)
		tmp = tmp.rename(columns={0: x})
		training_inputs = training_inputs.join(tmp)
	training_inputs = training_inputs.join(rolling_over_std)
	training_inputs = training_inputs.dropna(subset=["SPY"])
	training_inputs = training_inputs.dropna(subset=["Num STDS"])
	training_inputs = training_inputs.ix[:,1:]
	
	print(training_inputs)
	weights = create_weights(training_inputs) # start all weights at 0
	
	train(training_inputs, weights, expected)
	
	test_results(weights)
	print(weights)


if __name__ == "__main__":
	test_run()