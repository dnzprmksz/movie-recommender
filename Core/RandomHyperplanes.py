import numpy as np
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix


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
	dimension = signature.shape[1]

	# Calculate the number of columns for each band.
	if dimension % num_bands == 0:
		band_size = dimension / num_bands
	else:
		raise ValueError("Dimension of the signature matrix is not divisible by given number of bands.")

	# Initialize empty dictionary for hashmap.
	hashmap = {}

	loader = load("../Files/NormalizedUtilityMatrixCSR.npz")
	n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# Hash each band and assign to hash buckets.
	for user in remove_duplicates(n_utility_csr.tocoo().row):
		for band in xrange(0, num_bands):
			band_low = band * band_size
			band_high = band_low + band_size
			value = signature[user, band_low:band_high]
			key = binary_hash(value)
			hashmap.setdefault(key, set()).add(user)

	# Find and return the candidate pairs.
	pairs = [list(value) for value in hashmap.itervalues() if len(value) > 1]
	return pairs


def locality_sensitive_hashing_movie(signature, num_bands):
	num_movies = signature.shape[0]

	# Calculate the number of columns for each band.
	if num_movies % num_bands == 0:
		band_size = num_movies / num_bands
	else:
		raise ValueError("Dimension of the signature matrix is not divisible by given number of bands.")

	# Initialize empty dictionary for hashmap.
	hashmap = {}
	keys = []
	pairs = []

	loader = load("../Files/NormalizedUtilityMatrixCSC.npz")
	n_utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# Hash each band and assign to hash buckets.
	for movie in remove_duplicates(n_utility_csc.tocoo().col):
		for band in xrange(0, num_bands):
			band_low = band * band_size
			band_high = band_low + band_size
			value = signature[movie, band_low:band_high]
			key = binary_hash(value)
			hashmap.setdefault(key, set()).add(movie)  # Store (key, movie_id) in map.

	# Find and return the candidate pairs.
	pairs = [list(value) for value in hashmap.itervalues() if len(value) > 1]
	return pairs
