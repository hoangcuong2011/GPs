import numpy as np
import matplotlib.pyplot as pl

# Test data
n = 4
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# Define Squared Exponential Kernel (delta = 1)
# K_{SE}(a, b) = exp(-0.5 * (a - b) ^ 2)

def SquaredExponentialKernel(a, b, param):
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	print(sqdist)
	print((np.sum(a, 1) - b)**2)
	return np.exp(-.5 * (1/param) * sqdist)


def MyModifiedSquaredExponentialKernel(a, b, param):
	sqdist = (np.sum(a, 1) - b)**2
	return np.exp(-.5 * (1/param) * sqdist)


# see this for a reference: http://www.cs.toronto.edu/~duvenaud/cookbook/index.html
def RationalQuadraticKernel(a, b, param, alpha):
	# print(a)
	# print(np.sum(a**2,axis=1))
	# print(np.sum(a**2,axis=1).reshape(-1,1))
	# print(b)
	# print(b.T)
	# print(np.dot(a, b.T))
	# print(np.subtract(a, b)**2)
	sqdist = (np.sum(a, 1) - b)**2
	return ((-.5 * (1/param) / alpha * sqdist) + 1)**-alpha

def LinearKernel(a, b):
	# print(a)
	# print(np.sum(a**2,axis=1))
	# print(np.sum(a**2,axis=1).reshape(-1,1))
	# print(b)
	# print(b.T)
	# print(np.dot(a, b.T))
	# print(np.subtract(a, b)**2)
	sqdist = (np.sum(a, 1) * b)
	print(a)
	print(b)
	print(sqdist)
	return sqdist


param = 0.1
# K_ss = MyModifiedSquaredExponentialKernel(Xtest, Xtest, param)
#alpha = 2
K_ss = RationalQuadraticKernel(Xtest, Xtest, param, 2)

K_ss = LinearKernel(Xtest, Xtest)
# Get Cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample 3 sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
f_prior = np.dot(L, np.random.normal(size=(n,3)))

# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP prior')
pl.show()
