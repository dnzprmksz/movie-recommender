import numpy as np
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix


def binary_hash(value):
	key = 0
	for index in xrange(0, len(value)):
		if value[index]:
			key += 2 ** index
	return key

def remove_duplicates(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def locality_sensitive_hashing(signature, num_bands):
	num_users = signature.shape[0]
	dimension = signature.shape[1]

	# Calculate the number of columns for each band.
	if dimension % num_bands == 0:
		band_size = dimension / num_bands
	else:
		raise ValueError("Dimension of the signature matrix is not divisible by given number of bands.")

	# Initialize empty dictionary for hashmap.
	hashmap = {}
	keys = []
	pairs = []

	loader = load("Files/NormalizedUtilityMatrixCSR.npz")
	n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# Hash each band and assign to hash buckets.
	for user in remove_duplicates(n_utility_csr.tocoo().row):
		for band in xrange(0, num_bands):
			band_low = band * band_size
			band_high = band_low + band_size
			value = signature[user, band_low:band_high]
			key = binary_hash(value)
			hashmap.setdefault(key, set()).add(user)
			# Store (key, user_id) in map.

	# Find and return the candidate pairs.
	pairs = [list(value) for value in hashmap.itervalues() if len(value) > 1]
	# Below code left out for debugging purposes just in case keys are important
	# pairs = []
	# keys = []
	# for key, value in hashmap.iteritems():
	#	if len(value) > 1:
	#		keys.append(key)
	#		pairs.append(list(value))

	return pairs

def locality_sensitive_hashing_movie(signature, num_bands):
    num_movies = signature.shape[0]
    dimension = signature.shape[1]

    # Calculate the number of columns for each band.
    if num_movies % num_bands == 0:
	band_size = num_movies / num_bands
    else:
	raise ValueError("Dimension of the signature matrix is not divisible by given number of bands.")

    # Initialize empty dictionary for hashmap.
    hashmap = {}
    keys = []
    pairs = []

    loader = load("Files/NormalizedUtilityMatrixCSC.npz")
    n_utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

    # Hash each band and assign to hash buckets.
    for movie in xrange(1, num_movies):
	if len(n_utility_csc.getcol(movie).nonzero()[0]) > 0:
	    for band in xrange(0, num_bands):
		band_low = band * band_size
		band_high = band_low + band_size
		value = signature[movie, band_low:band_high]
		key = binary_hash(value)
		hashmap.setdefault(key, []).append(movie)  # Store (key, movie_id) in map.

    # Find and return the candidate pairs.
    for key, value in hashmap.iteritems():
	if len(value) > 1:
	    keys.append(key)
	    pairs.append(value)

    return keys, pairs


def calculate_user_similarity(user_id, candidate_id, signature):
    dimension = signature.shape[1]
    u = signature[user_id]
    v = signature[candidate_id]
    common = 0

    # Find the number of common values.
    for i in xrange(0, dimension):
	if u[i] == v[i]:
	    common += 1

    # Find the Jaccard similarity of two vectors. Calculate the angle and distance between them.
    similarity = float(common) / dimension
    angle = 180 * (1 - similarity)
    distance = 1 - np.cos(np.deg2rad(angle))
    return angle, distance


# Min-hashing with random vectors. More vectors produce better approximation.
def generate_user_signature(num_vectors=120):
    # Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
    loader = load("Files/NormalizedUtilityMatrixCSR.npz")
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
		signature[i, j] = np.random.randint(2)	# Convert 0s into 0 or 1. New 0s will be considered as negative.
    signature[signature > 0] = 1
    signature[signature <= 0] = -1

    # Move signature matrix to a boolean matrix to compress the size.
    compressed = np.full((num_users, num_vectors), False, dtype=bool)
    compressed[signature == 1] = True

    # Save signature matrix.
    np.save("Files/UserSignature", compressed)
    np.save("Files/UserSignatureInteger", signature)


# Min-hashing with random vectors. More vectors produce better approximation.
def generate_movie_signature(num_vectors=240):
    # Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
    loader = load("Files/NormalizedUtilityMatrixCSC.npz")
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
		signature[i, j] = np.random.randint(2)	# Convert 0s into 0 or 1. New 0s will be considered as negative.
    signature[signature > 0] = 1
    signature[signature <= 0] = -1

    # Move signature matrix to a boolean matrix to compress the size.
    compressed = np.full((num_movies, num_vectors), False, dtype=bool)

    # Save signature matrix.
    np.save("Files/MovieSignature", compressed)
