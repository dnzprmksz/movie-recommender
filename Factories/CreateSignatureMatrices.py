import numpy as np
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix


# Min-hashing with random vectors. More vectors produce better approximation.
def generate_user_signature(num_vectors=120):
	# Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
	loader = load("../Files/NormalizedUtilityMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Get number of movies as the dimensions of the vector space.
	dimension = utility_csr.shape[1]
	num_users = utility_csr.shape[0]
	
	# Generate random vectors with values between +1 and -1, and form a vertical matrix with them.
	vectors = 2 * np.random.rand(dimension, num_vectors) - 1
	
	# Initialize signature matrix with 0s.
	signature = np.zeros((num_users, num_vectors))
	
	# Apply Random Hyperplanes and Cosine Distance similarity algorithm on utility matrix.
	for user_id in xrange(1, num_users):
		rating = utility_csr.getrow(user_id)
		descriptor = rating.dot(vectors)
		signature[user_id] = descriptor
	
	# Post process signature matrix to have only +1/-1 values.
	for i in xrange(0, signature.shape[0]):
		for j in xrange(0, signature.shape[1]):
			if signature[i, j] == 0:
				signature[i, j] = np.random.randint(2)  # Convert 0s into 0 or 1. New 0s will be considered as negative.
	
	# Label positives as +1 and negatives as -1.
	signature[signature > 0] = 1
	signature[signature <= 0] = -1
	
	# Move signature matrix to a boolean matrix to compress the size.
	compressed = np.full((num_users, num_vectors), False, dtype=bool)
	compressed[signature == 1] = True
	
	# Save signature matrix.
	np.save("../Files/UserSignature", compressed)


# Min-hashing with random vectors. More vectors produce better approximation.
def generate_movie_signature(num_vectors=240):
	# Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
	loader = load("../Files/NormalizedUtilityMatrixCSC.npz")
	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Get number of users as the dimensions of the vector space.
	dimension = utility_csc.shape[0]
	num_movies = utility_csc.shape[1]
	
	# Generate random vectors with values between +1 and -1, and form a vertical matrix with them.
	vectors = 2 * np.random.rand(dimension, num_vectors) - 1
	
	# Initialize signature matrix with 0s.
	signature = np.zeros((num_movies, num_vectors), np.int8)
	
	# Apply Random Hyperplanes and Cosine Distance similarity algorithm on utility matrix.
	for movie_id in xrange(1, num_movies):
		rating = np.transpose(utility_csc.getcol(movie_id))
		descriptor = rating.dot(vectors)
		signature[movie_id] = descriptor
	
	# Post process signature matrix to have only +1/-1 values.
	for i in xrange(0, signature.shape[0]):
		for j in xrange(0, signature.shape[1]):
			if signature[i, j] == 0:
				signature[i, j] = np.random.randint(2)  # Convert 0s into 0 or 1. New 0s will be considered as negative.
	
	# Label positives as +1 and negatives as -1.
	signature[signature > 0] = 1
	signature[signature <= 0] = -1
	
	# Move signature matrix to a boolean matrix to compress the size.
	compressed = np.full((num_movies, num_vectors), False, dtype=bool)
	
	# Save signature matrix.
	np.save("../Files/MovieSignature", compressed)
