import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('data/housing.csv')

X, y = df.GrLiveArea, df.SalePrice

# Predict Insurance.

# Calculate Mean and Variance.
def mean(series):
	N = len(series)
	return sum(series) / N

def variance(series):
	N = len(series)
	return sum([(series[i] - mean(series))**2 for i in range(N)])

# Calculate Covariance.
def covariance(series_x, series_y):
	if len(series_x == series_y):
		N = len(series_x)
		return sum([(series_x[i] - mean(series_x)) * (series_y[i] - mean(series_y)) for i in range(N)])
	else:
		raise IndexError("Sizes must be same!!")

# Estimate Coefficients.
def coefficients(X, Y):
	beta_1 = covariance(X, Y) / variance(X)
	beta_0 = mean(Y) - beta_1 * mean(X)
	return beta_0, beta_1

# Make Predictions.
def prediction(X, coefficients):
	beta_0, beta_1 = coefficients
	return [beta_0 + beta_1 * x for x in X]

y_hat = prediction(X, coefficients(X, Y))

# Evaluate with r^2
def rsqr(Y, y_hat):
	N = len(Y)
	ss_tot = sum([(Y[i] - mean(Y))**2 for i in range(N)])
	ss_res = sum([(Y[i] - y_hat[i])**2 for i in range(N)])
	return 1 - ss_res/ss_tot

# Plot
plt.scatter(X, Y)
plt.plot(X, y_hat)
plt.show()

# Summary
