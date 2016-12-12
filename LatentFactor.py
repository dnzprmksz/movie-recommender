import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def estimate_user_rating(user_id, movie_id):
	p = np.load("SGD_P.npy")
	q = np.load("SGD_Q.npy")
	return q[movie_id].dot(np.transpose(p[user_id]))


def create_factor_matrices(u1, u2, lambda1, lambda2, factor_count, num_iterations):
	start_time = time()
	
	# Load the utility matrix which will be used for the calculation of latent factor vectors.
	loader = load("UtilityMatrix1000CSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_users = utility_csr.shape[0]
	
	# Apply Singular Value Decomposition (SVD) on utility matrix to approximate P and Q matrices of latent factor.
	u, s, v = svds(utility_csr.asfptype())
	p = u[:, 1:factor_count+1]  # User factor matrix as row vector.
	q = v[1:factor_count+1, :]  # Item factor matrix as column vector.
	q = np.transpose(q)         # Convert item factor matrix into row vector.
	
	# Stochastic Gradient Descent
	for i in xrange(1, num_iterations):
		movie_index = 0
		for user_id in xrange(1, num_users - 1):
			row = utility_csr.getrow(user_id)
			for movie_id in row.indices:
				rating = utility_csr.data[movie_index]  # r.xi in the formula.
				pc = np.transpose(p[user_id])  # P as column vector.
				pr = p[user_id]   # P as row vector.
				qr = q[movie_id]  # Q as row vector.
				
				# One iteration of update part of Stochastic Gradient Descent algorithm.
				error = rating - qr * pc
				q[movie_id] += np.multiply(u1, (np.multiply(error, pr) - np.multiply(lambda2, qr)))
				p[user_id] += np.multiply(u2, (np.multiply(error, qr) - np.multiply(lambda1, pr)))
				movie_index += 1
	
	# Save P and Q matrices.
	np.save("SGD_P", p)
	np.save("SGD_Q", q)
	print "%f seconds elapsed." % (time() - start_time)
