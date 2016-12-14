import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from LatentFactor import estimate_user_rating
from RandomHyperplanes import generate_user_signature, locality_sensitive_hashing, __calculate_similarity_DO_NOT_USE__


def test_latent_factor(user_id, movie_id):
	ranking = estimate_user_rating(user_id, movie_id)
	print("User %d Movie %d. Estimated ranking: %d" % (user_id, movie_id, ranking))
	

def test_lsh(num_bands):
	start_time = time()
	signature = np.load("UserSignature.npy")
	pairs = locality_sensitive_hashing(signature, num_bands)
	print("---")
	
	loader = load("NormalizedUtilityMatrixCSR.npz")
	n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	distances = set()
	for pair in pairs:
		print(pair)
		num = len(pair)
		for i in xrange(0, num):
			for j in xrange(i+1, num):
				u = n_utility_csr.getrow(pair[i]).toarray()
				v = n_utility_csr.getrow(pair[j]).toarray()
				distances.add(cosine(u, v))
	
	print("---")
	print("Distances: %d" % len(distances))
	count = 0
	for i in distances:
		if i <= 0.5:
			count += 1
	print("Distances less than 0.50: %d" % count)
	print(sorted(distances))


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
		angle, distance = __calculate_similarity_DO_NOT_USE__(i, j, signature)
		
		print "User %d and Candidate %d" % (i, j)
		print "Angle and Distance: %d degrees, %f" % (angle, distance)
		print "Original Distance:  %f\n" % cosine(u, v)

	print "\n%f seconds elapsed." % (time() - start_time)

# Test cases.
#test_random_hyperplanes_similarity()
test_lsh(4)
#test_latent_factor(777, 731)
#test_latent_factor(777, 2027)
#test_latent_factor(777, 11127)
