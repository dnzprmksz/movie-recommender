import numpy as np
from time import time
from numpy import load, mean
from scipy.sparse import csr_matrix, csc_matrix

start_time = time()

loader = load("UtilityMatrixCSC.npz")
utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

loader = load("UtilityMatrixCSR.npz")
utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

num_movies = utility_csc.shape[1]
num_users = utility_csr.shape[1]

# Normalize CSC matrix for item-item collaborative filtering usage.
for index in range(1, num_movies):
	ratings = utility_csc.getcol(index)
	if ratings.nnz > 0:  # Process the column if it is non-empty.
		average = mean(ratings.data)  # Average rating of the movie.
		utility_csc.data[utility_csc.indptr[index]:utility_csc.indptr[index+1]] -= average  # Normalize each rating of movie.

# Normalize CSR matrix for user-user collaborative filtering usage.
for index in range(1, num_users):
	ratings = utility_csr.getrow(index)
	if ratings.nnz > 0:  # Process the row if it is non-empty.
		average = mean(ratings.data)  # Average rating of the user.
		utility_csr.data[utility_csr.indptr[index]:utility_csr.indptr[index+1]] -= average  # Normalize each rating of user.

# Save normalized utility matrices.
np.savez("NormalizedUtilityMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr, shape=utility_csc.shape)
np.savez("NormalizedUtilityMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr, shape=utility_csr.shape)

print "%f seconds elapsed." % (time() - start_time)
