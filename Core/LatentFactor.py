import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix


def estimate_user_rating(user_id, movie_id, p, q):
	return q[movie_id].dot(np.transpose(p[user_id]))


def create_factor_matrices(u1, u2, lambda1, lambda2, factor_count):
	start_time = time()

	# Load the utility matrix which will be used for the calculation of latent factor vectors.
	loader = load("../Files/TrainingCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_users = utility_csr.shape[0]
	num_movies = utility_csr.shape[1]
	utility_csr = utility_csr.asfptype()  # Change data type into float.

	# Approximate P and Q matrices of latent factor.
	p = 7 * np.random.rand(num_users, factor_count) + 3  # User factor matrix as row vector.
	q = 7 * np.random.rand(num_movies, factor_count) + 3  # Item factor matrix as row vector.

	loop_count = 0
	# Stochastic Gradient Descent
	while True:
		loop_start_time = time()
		loop_count += 1
		for user_id in xrange(1, num_users):
			row = utility_csr[user_id]
			movie_index = 0  # Use individual movie index in user's row data set, for optimization.
			for movie_id in row.indices:
				rating = row.data[movie_index]  # r.xi in the formula.
				pc = np.transpose(p[user_id])  # P as column vector.
				pr = p[user_id]   # P as row vector.
				qr = q[movie_id]  # Q as row vector.

				# One iteration of update part of Stochastic Gradient Descent algorithm.
				error = rating - qr.dot(pc)
				q[movie_id] += u1 * (error * pr - lambda2 * qr)
				p[user_id] += u2 * (error * qr - lambda1 * pr)
				movie_index += 1

		# Print running time analysis for monitoring the status and performance of the job.
		loop_time = time() - loop_start_time
		total_time = time() - start_time
		print "Finished loop #%d. Took %d minutes %d seconds." % (loop_count, loop_time / 60, loop_time % 60)
		print "Total running time is %d hours %d minutes.\n" % (total_time / 3600, (total_time / 60) % 60)

		# Save matrices in every 25 loop.
		if loop_count % 25 == 0:
			p_filename = "../Files/SGD_P_" + str(loop_count)
			q_filename = "../Files/SGD_Q_" + str(loop_count)
			np.save(p_filename, p)
			np.save(q_filename, q)

# Create P and Q matrices. Comment out below function or call in another file.
# create_factor_matrices(0.0001, 0.0001, 0.001, 0.001, 3
