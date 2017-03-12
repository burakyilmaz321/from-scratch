import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def costFunction(theta, X, y):
	"""
	Returns cost and gradient
	"""
	m = len(y)
	J = (1/m)*sum(-y * np.log(sigmoid(np.dot(X,theta))) - (1-y) * np.log(1-sigmoid(np.dot(X,theta))))
	grad = (1/m) * np.dot(X.T, (sigmoid(np.dot(X,theta)) - y))
	return J, grad

def main():	
	# Import data
	data = np.loadtxt('data/logit.txt', delimiter=',')
	X, y = data[:, [0, 1]], data[:, 2]
	m, n = X.shape

	# Setup the data matrix
	X = np.concatenate([np.ones((m, 1)), X], axis=1)

	# Initialize fitting parameters
	initial_theta = np.ones((n+1, 1))

	# Compute initial cost and gradient
	costFunction(initial_theta, X, y)

if __name__ == '__main__':
	main()
