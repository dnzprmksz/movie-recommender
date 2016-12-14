from __future__ import print_function
import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from RandomHyperplanes import calculate_similarity, generate_user_signature, locality_sensitive_hashing


def test_lsh():
	start_time = time()
	signature = np.load("UserSignature.npy")
	pairs = locality_sensitive_hashing(signature[0:10000], 4)
	print("\n%f seconds elapsed." % (time() - start_time))


def rmse(acquired_data, test_data):
	diff = acquired_data - test_data
	out = np.sqrt(diff**2/len(test_data.data))
	return out


def test_random_hyperplanes_similarity(i=62500, regenerate=False, vector_count=125):
	start_time = time()
	
	if regenerate:
		generate_user_signature(vector_count)
	
	# Load necessary matrices.
	loader = load("UtilityMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	signature = np.load("UserSignature.npy")
	
	for j in [1, 2, 62500]:
		u = utility_csr.getrow(i).toarray()
		v = utility_csr.getrow(j).toarray()
		angle, distance = calculate_similarity(i, j, signature)
		
		print("User %d and Candidate %d" % (i, j))
		print("Angle and Distance: %d degrees, %f" % (angle, distance))
		print("Original Distance:  %f\n" % cosine(u, v))

	print("\n%f seconds elapsed." % (time() - start_time))

# Test cases.
#test_random_hyperplanes_similarity()
test_lsh()
