import numpy as np
import matplotlib.pyplot as pl

# Test data
n = 1200
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# Define the kernel function
def MyModifiedSquaredExponentialKernel(a, b, param):
	# print(a)
	# print(b)
	sqdist = (np.sum(a, 1).reshape(-1,1) - np.sum(b,1))**2
	return np.exp(-.5 * (1/param) * sqdist)

param = 1
K_testtest = MyModifiedSquaredExponentialKernel(Xtest, Xtest, param)

# Get cholesky decomposition (square root) of the
# covariance matrix
# L = np.linalg.cholesky(K_testtest + 1e-15*np.eye(n))

# Noiseless training data

Xtrain = np.linspace(-5, 5, n/10).reshape(-1,1)
ytrain = np.sin(Xtrain)

# Apply the kernel function to our training points - note we plus some noise to make sure K_traintrain positive definite
K_traintrain = MyModifiedSquaredExponentialKernel(Xtrain, Xtrain, param)+ 0.0001 * np.eye(len(ytrain))

#L L^T = K_traintrain
L_traintrain = np.linalg.cholesky(K_traintrain + 0.00005*np.eye(len(Xtrain)))

# Compute the mean at our test points.
K_traintest = MyModifiedSquaredExponentialKernel(Xtrain, Xtest, param)
K_testtrain = MyModifiedSquaredExponentialKernel(Xtest, Xtrain, param)

# AX = B -> X = A^-1B
# L_traintrain Lk = K_traintest -> Lk = L_traintrain^-1 K_traintest
Lk = np.linalg.solve(L_traintrain, K_traintest)
print(Lk)


# (L_traintrain^-1 K_traintest)^T L_traintrain^-1 ytrain
mu = np.dot(Lk.T, np.linalg.solve(L_traintrain, ytrain)).reshape((n,))

# print(K_traintest.shape)
# print(K_traintrain.shape)
# print(K_testtest.shape)
# print(K_testtrain.shape)
#print(np.linalg.inv(K_traintrain).shape)
#print(ytrain.shape)

#mu_2 = np.dot(K_traintest, np.linalg.solve(K_traintrain, ytrain))
mu = np.dot(np.dot(K_testtrain, np.linalg.inv(K_traintrain)), ytrain)

# print((np.dot(K_traintest.T, np.linalg.inv(K_traintrain))).shape)
SIGMA = K_testtest - np.dot(np.dot(K_testtrain, np.linalg.inv(K_traintrain)), K_testtrain.T)
# print(SIGMA.shape)
#print(K_testtest.shape)
# print(mu_2)
# print(mu)

# Compute the standard deviation so we can plot it
# s2 = np.diag(K_testtest) - np.sum(Lk**2, axis=0)
# stdv = np.sqrt(s2)
# Draw samples from the posterior at our test points.
L = np.linalg.cholesky(SIGMA + 1e-6*np.eye(n))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,1000)))

pl.plot(Xtrain, ytrain, 'bs', ms=8)
pl.plot(Xtest, f_post)
# pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
# pl.plot(Xtest, mu, 'r--', lw=2)
pl.axis([-5, 5, -3, 3])
pl.title('Three samples from the GP posterior')
pl.show()
