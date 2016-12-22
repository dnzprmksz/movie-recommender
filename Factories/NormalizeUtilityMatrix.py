import numpy as np
from time import time
from numpy import load, mean
from scipy.sparse import csr_matrix


def normalize_utility_matrix():
	start_time = time()

	loader = load("../Files/TrainingMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	num_users = utility_csr.shape[0]

	# Normalize CSR matrix for user-user collaborative filtering usage.
	for index in xrange(1, num_users):
		ratings = utility_csr[index]
		if len(ratings.data) > 0:  # Process the row if it is non-empty.
			average = mean(ratings.data)  # Average rating of the user.
			utility_csr.data[utility_csr.indptr[index]:utility_csr.indptr[index+1]] -= int(average)	 # Normalize each rating of user.

	utility_csr.eliminate_zeros()
	utility_csc = utility_csr.tocsc()
	# Save normalized utility matrices.
	np.savez("../Files/NormalizedUtilityMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr, shape=utility_csc.shape)
	np.savez("../Files/NormalizedUtilityMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr, shape=utility_csr.shape)

	print "%f seconds elapsed." % (time() - start_time)

normalize_utility_matrix()