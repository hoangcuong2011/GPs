import numpy as np
import matplotlib.pyplot as pl

# Test data
n = 40
Xtest = np.linspace(0, 10, n).reshape(-1,1)

# good reference: http://www.cs.ubc.ca/~nando/540-2013/lectures/gp.py
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

# the most famous GP? http://www0.cs.ucl.ac.uk/staff/J.Shawe-Taylor/courses/ATML-1.pdf
def BrownianMotion(a, b):
	sqdist = np.minimum(np.sum(a, 1), b)
	print(sqdist)
	return sqdist

def fixedSomePoints(f_prior):
	temp = f_prior.ravel()
	print(f_prior)

	print("------------------")
	# print(temp)
	print("------------------")
	count = 0
	line = 0
	for y in np.nditer(temp, op_flags=['readwrite']):
		if (line+1)%5==0:
			# print(count, "it happens")
			y[...] = line/ float(10)
		count = count + 1
		if count == dataPoints :
			count = 0
			line = line + 1
	# print(temp)
	print("------------------")
	B = np.reshape(temp, (-1, dataPoints))
	print(B)

	print("------------------")

	print(Xtest)
	return B

param = 0.25
K_ss = MyModifiedSquaredExponentialKernel(Xtest, Xtest, param)
#alpha = 2
# K_ss = RationalQuadraticKernel(Xtest, Xtest, param, 2)

# K_ss = BrownianMotion(Xtest, Xtest)
# Get Cholesky decomposition (square root) of the
# covariance matrix
L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))
# Sample sets of standard normals for our test points,
# multiply them by the square root of the covariance matrix
dataPoints = 100

f_prior = np.dot(L, np.random.normal(size=(n,dataPoints)))


# print(f_prior)

# f_prior = fixedSomePoints(f_prior) #I assume we have data for this

# print(f_prior)

print(Xtest)


# Now let's plot the 3 sampled functions.
pl.plot(Xtest, f_prior)
pl.axis([0, 10, -5, 5])
pl.title('Samples from the GP prior')
pl.show()
