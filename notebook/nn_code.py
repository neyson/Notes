# basic_sigmoid
import math
def basic_sigmoid(x):
	"""
	Compute sigmoid of x.
	
	Arguments:
	x -- A scalar
	
	Return:
	s -- sigmoid(x)
	"""	
	s = 1/(1+math.exp(-x))	
	return s
	
# sigmoid
import numpy as np
def sigmoid(x):
	"""
	Compute the sigmoid of x
	
	Arguments:
	x -- A scalar or numpy array of any size
	
	Return:
	s -- sigmoid(x)
	"""
	s = 1/(1+np.exp(-x))
	return s
	
# sigmoid_derivative
def sigmoid_derivative(x):
	"""
	Compute the gradient (also called the slope or derivative) of the sigmoid function with repect to its input x. You can store the output of the sigmoid function into variables and then use it to caculate the gradient.
	
	Arguments:
	x -- A scalar or numpy array
	
	Return:
	ds -- Your computed gradient.
	"""
	s = 1 / (1 + np.exp(-x))
	ds = s * (1 - s)
	return ds
	
# image2vector
def image2vector(image):
	"""
	Arguments:
	image -- a numpy array of shape (length, height, depth)
	
	Returns:
	v -- a vector of shape (length * height * depth, 1)
	"""
	v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
	return v
	
# normalizeRows
def normalizeRows(x):
	"""
	Implement a function that normalizes each row of the matrix x (to have unit length).
	
	Arguments:
	x -- A numpy matrix of shape(n, m)
	
	Returns:
	x -- The normalized(by row) numpy matrix. You are allowed to modify x.
	"""
	# Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
	x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
	
	# Divide x by its norm.
	x = x / x_norm
	
	return x
	
# softmax
def softmax(x):
	"""
	Calculates the softmax for each row of the input x.
	Your code should work for a row vector and also for matrices of shape (n, m).
	
	Arguments:
	x -- A numpy matrix of shape (n, m)
	
	Returns:
	s -- A numpy matrix equal to the softmax of x, of shape (n, m)
	"""
	# Apply exp() element-wise to x. Use np.exp(...).
	x_exp = np.exp(x)
	
	# Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis=1, keepdims=True).
	x_sum = np.sum(x_exp, axis=1, keepdims=True)
	
	# Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
	s = x_exp / x_sum
	return s
	
# L1
def L1(yhat, y):
	"""
	Arguments:
	yhat -- vector of size m (predicted labels)
	y -- vector of size m (true labels)
	
	Returns:
	loss -- the value of the L1 loss function.
	"""
	loss = sum(abs(yhat - y))
	return loss 
	
# L2
def L2(yhat, y):
	"""
	Arguments:
	yhat -- vector of size m (predicted labels)
	y -- vector of size m (true labels)
	
	Returns:
	loss -- the value of the L2 loss function defined above
	"""
	loss = np.dot(yhat-y, yhat-y)
	return loss 
	
