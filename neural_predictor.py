import os
import numpy as np
import util
import pandas as pd 

learning_rate = 0.0001


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
	#print(inputs.ix[index][2])
	output = inputs.ix[index].multiply(weights.as_matrix()).sum()
	output = 0
	return 1.0 if output >= 0.0 else 0.0


def train(inputs, weights, expected):
	print("Training...\n")
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
	return expected

# Get the U.S Dollar / Euro
def input_dollar_over_euro(dates):
	df = pd.DataFrame(index=dates)
	df = pd.read_csv(
			os.path.join("data", "DEXUSEU.csv"),
			index_col="DATE",
			parse_dates=True,
			usecols=["DATE", "DEXUSEU"],
			na_values=['nan'],
			engine='python'
		)
	df = df.applymap(lambda x: np.nan if (x == '.') else x) # Set . as NaN
	df = df.applymap(lambda x: float(x)) # convert strings to floats
	df = df.bfill()
	return df		

def input_distance_from_upper_and_lower_bollinger_band(df):
	# number of std deviations away from rolling mean
	upper_band, lower_band = util.get_bollinger_bands(df)
	distance_from_upper = upper_band - df
	distance_from_lower = lower_band - df

	distance_from_upper = distance_from_upper.rename(columns={'SPY': 'Distance From Upper'})
	distance_from_lower = distance_from_lower.rename(columns={'SPY': 'Distance From Lower'})

	return distance_from_upper, distance_from_lower
	
def test_results(weights):
	print("Testing with weights...\n")
	start_date = '2014-01-02'
	end_date = '2016-08-01'
	dates = pd.date_range(start_date, end_date)
	spy = util.get_data(['SPY'], dates)
	normed_spy = util.normalize_data(spy)
	expected = get_expected(dates)
	correct_count = 0
	test_inputs = create_inputs(dates, training=False)
	# Test the output
	for date, row in test_inputs.iterrows():
		if(expected.ix[date][0] == output(date, test_inputs, weights)):
			correct_count += 1

	print("Correct count: {}, Total: {}, Percent Correct: {}".format(correct_count, str(test_inputs.shape[0]), str((correct_count/test_inputs.shape[0])*100) ))
		#rolling_mean_over_std = rolling_over_std.ix[date][0]
	

def create_inputs(dates, training=True):
	inputs = pd.DataFrame(index=dates)
	spy = util.get_data(['SPY'], dates)
	normed_spy = util.normalize_data(spy) # SPY Normalized
	inputs = inputs.join(spy)

	#Set the inputs
	input_distance_from_upper, input_distance_from_lower = input_distance_from_upper_and_lower_bollinger_band(normed_spy)

	dollar_over_euro = input_dollar_over_euro(dates) # dollar / euro

	if training:
		input_distance_from_upper.shift(1)
		input_distance_from_lower.shift(1)
		dollar_over_euro.shift(1)

	inputs = inputs.join(input_distance_from_upper)
	inputs = inputs.join(input_distance_from_lower)
	inputs = inputs.join(dollar_over_euro)

	# Join inputs together
	inputs = inputs.dropna(subset=["SPY"])
	inputs = inputs.dropna(subset=["Distance From Upper"])
	inputs = inputs.dropna(subset=["Distance From Lower"])
	inputs = inputs.dropna(subset=["DEXUSEU"])
	inputs = inputs.ix[:,1:]
	return inputs


def test_run():
	start_date = '1993-08-31'
	end_date = 	'2014-01-01'
	dates = pd.date_range(start_date, end_date)
	
	expected = get_expected(dates) # From SPY
	training_inputs = create_inputs(dates)

	weights = create_weights(training_inputs) # start all weights at 0
	
	train(training_inputs, weights, expected)
	#weights = pd.Series([-0.000609, 0.001257])
	print("Weights: \n", weights, "\n")

	test_results(weights)


if __name__ == "__main__":
	test_run()