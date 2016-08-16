import numpy as np
import time

def get_max_index(a):
	return np.argmax(a)

def test_run():
	start = time.time()
	a = np.array([9, 6, 2, 3, 12, 14, 7, 10], dtype=np.int32)  # 32-bit integer array
	print("Array:", a)
	# Find the maximum and its index in array
	print("Index of max.:", get_max_index(a))
	print("Maximum value:", a.max())
	end = time.time()
	print("The time taken by the print statement is ", start - end, " seconds.")
	#Get every element of the array less than the mean
	print("Every element less than mean", a[a>a.mean()])
	print("Multiply every element by 2nd power", a ** 2)

if __name__ == "__main__":
	test_run()