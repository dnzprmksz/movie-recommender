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
	loader = load("UtilityMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_users = utility_csr.shape[0]
	utility_csr = utility_csr.asfptype()
	
	# Apply Singular Value Decomposition (SVD) on utility matrix to approximate P and Q matrices of latent factor.
	u, s, v = svds(utility_csr)
	p = u[:, 1:factor_count+1]  # User factor matrix as row vector.
	q = v[1:factor_count+1, :]  # Item factor matrix as column vector.
	q = np.transpose(q)         # Convert item factor matrix into row vector.
	
	# Stochastic Gradient Descent
	for i in xrange(1, num_iterations):
		loop_start_time = time()
		movie_index = 0
		for user_id in xrange(1, num_users - 1):
			row = utility_csr.getrow(user_id)
			for movie_id in row.indices:
				rating = utility_csr.data[movie_index]  # r.xi in the formula.
				pc = np.transpose(p[user_id])  # P as column vector.
				pr = p[user_id]   # P as row vector.
				qr = q[movie_id]  # Q as row vector.
				
				# TODO: Rewrite mathematical formula for user/item utility matrix and check this implementation.
				# One iteration of update part of Stochastic Gradient Descent algorithm.
				error = rating - qr.dot(pc)
				q[movie_id] += u1 * (error * pr - lambda2 * qr)
				p[user_id] += u2 * (error * qr - lambda1 * pr)
				movie_index += 1
		# Print running time analysis for monitoring the status and performance of the job.
		loop_time = time() - loop_start_time
		total_time = time() - start_time
		estimated_total = num_iterations * loop_time
		estimated_remaining = (num_iterations - i) * loop_time
		print "Finished loop #%d. Took %d minutes %d seconds." % (i, loop_time/60, loop_time % 60)
		print "Total running time is %d hours %d minutes." % (total_time/3600, (total_time/60) % 60)
		print "Estimated total time is %d hours %d minutes." % (estimated_total/3600, (estimated_total/60) % 60)
		print "Estimated remaining time is %d hours %d minutes.\n" % (estimated_remaining/3600, (estimated_remaining/60) % 60)
	
	# Save P and Q matrices.
	np.save("SGD_P", p)
	np.save("SGD_Q", q)
	print "Finished in %f seconds." % (time() - start_time)
	print "Finished in %f minutes." % ((time() - start_time)/60)
	print "Finished in %f hours." % ((time() - start_time)/3600)

create_factor_matrices(0.0001, 0.0001, 0.001, 0.001, 2, 1000)
