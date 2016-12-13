import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix


def calculate_similarity(user_id, candidate_id):
	signature = np.load("UserSignature.npy")
	num_movies = signature.shape[1] - 1
	u = signature[user_id]
	v = signature[candidate_id]
	same = 0
	
	# Find the number of common values.
	for i in xrange(1, num_movies):
		if u[i] == v[i]:
			same += 1
	
	return float(same)/num_movies


# Number of random vector to calculate signature. More vectors produce better approximation.
def generate_user_signature(num_vectors=200):
	start_time = time()
	
	# Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
	loader = load("NormalizedUtilityMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Get number of movies as the dimensions of the vector space.
	dimension = utility_csr.shape[1]
	num_users = utility_csr.shape[0]
	
	# Generate 100 random vectors with +1/-1 values and form a vertical matrix with them.
	vectors = np.random.randint(2, size=(dimension, num_vectors))  # Create a random matrix with 0 and 1 values.
	vectors[vectors == 0] = -1  # Replace all 0s with -1s.
	
	# Initialize signature matrix with 0s.
	signature = np.zeros((num_users + 1, num_vectors), np.int8)
	
	# Apply Random Hyperplanes and Cosine Distance similarity algorithm on utility matrix.
	for user_id in xrange(1, num_users - 1):
		rating = utility_csr.getrow(user_id)
		descriptor = rating.dot(vectors)
		signature[user_id] = descriptor
		if user_id % 10000 == 0:
			print "%d/%d completed in %d seconds." % (user_id, num_users, time() - start_time)
	
	# Post process signature matrix to have only +1/-1 values.
	signature[signature == 0] = np.random.randint(2)  # Convert 0s into 0 or 1. New 0s will be considered as negative.
	signature[signature > 0] = 1
	signature[signature <= 0] = -1
	
	# Save signature matrix.
	np.save("UserSignature", signature)
	
	print "Finished in %d seconds." % (time() - start_time)
