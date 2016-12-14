import numpy as np
from numpy import load
from scipy.sparse import csr_matrix


def binary_hash(value):
	key = 0
	for index in xrange(0, len(value)):
		if value[index] == 1:
			key += 2**index
	return key


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
	pairs = []
	
	# Hash each band and assign to hash buckets.
	for band in xrange(0, num_bands):
		for user in xrange(1, num_users):
			band_low = band * band_size
			band_high = band_low + band_size
			value = signature[user, band_low:band_high]
			key = binary_hash(value)
			hashmap.setdefault(key, []).append(user)  # Store (key, user_id) in map.
	
	# Find and return the candidate pairs.
	for value in hashmap.itervalues():
		if len(value) > 1:
			a, d = calculate_similarity(value[0], value[1], signature)
			print value, a
			pairs.append(value)
	
	return pairs


def calculate_similarity(user_id, candidate_id, signature):
	dimension = signature.shape[1]
	u = signature[user_id]
	v = signature[candidate_id]
	common = 0
	
	# Find the number of common values.
	for i in xrange(0, dimension):
		if u[i] == v[i]:
			common += 1
	
	similarity = float(common)/dimension
	angle = 180 * (1 - similarity)
	distance = 1 - np.cos(np.deg2rad(angle))
	return angle, distance


# Min-hashing with random vectors. More vectors produce better approximation.
def generate_user_signature(num_vectors=100):
	# Using normalized utility matrix, since similarity in normalized matrix is more important than vanilla.
	loader = load("NormalizedUtilityMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Get number of movies as the dimensions of the vector space.
	dimension = utility_csr.shape[1]
	num_users = utility_csr.shape[0]
	
	# Generate random vectors with values between +1 and -1, and form a vertical matrix with them.
	vectors = 2 * np.random.rand(dimension, num_vectors) - 1
	
	# Initialize signature matrix with 0s.
	signature = np.zeros((num_users, num_vectors), np.int8)
	
	# Apply Random Hyperplanes and Cosine Distance similarity algorithm on utility matrix.
	for user_id in xrange(1, num_users):
		rating = utility_csr.getrow(user_id)
		descriptor = rating.dot(vectors)
		signature[user_id] = descriptor
	
	# Post process signature matrix to have only +1/-1 values.
	signature[signature == 0] = np.random.randint(2)  # Convert 0s into 0 or 1. New 0s will be considered as negative.
	signature[signature > 0] = 1
	signature[signature <= 0] = -1
	
	# Save signature matrix.
	np.save("UserSignature", signature)
