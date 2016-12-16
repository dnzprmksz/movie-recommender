import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial.distance import cosine

import Toolkit
from LatentFactor import estimate_user_rating
from RandomHyperplanes import generate_user_signature, locality_sensitive_hashing, __calculate_similarity_DO_NOT_USE__, \
    locality_sensitive_hashing_movie, generate_movie_signature


def test_latent_factor(user_id, movie_id):
<<<<<<< Updated upstream
	ranking = estimate_user_rating(user_id, movie_id)
	loader = load("Files/TrainingMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	real = utility_csr[user_id, movie_id]
	print "User %d Movie %d. Real ranking: %f. Estimated ranking: %f" % (user_id, movie_id, real, ranking)


def test_lsh_speed(num_bands):
	start_time = time()
	signature = np.load("Files/UserSignature.npy")
	keys, pairs = locality_sensitive_hashing(signature, num_bands)
	print "Finished in %d seconds." % (time() - start_time)


def test_lsh(num_bands):
	start_time = time()
	signature = np.load("Files/UserSignature.npy")
	pairs = locality_sensitive_hashing(signature, num_bands)
	print "LSH completed in %d seconds." % (time() - start_time)
	
	loader = load("Files/NormalizedUtilityMatrixCSR.npz")
	n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	size = len(pairs)
	count = 0
	distances = []
	for pair in pairs:
		num = len(pair)
		for i in xrange(0, num):
			u = n_utility_csr.getrow(pair[i]).toarray()
			for j in xrange(i+1, num):
				v = n_utility_csr.getrow(pair[j]).toarray()
				distances.append(cosine(u, v))
		count += 1
		if count % 500 == 0:
			print "%d/%d completed." % (count, size)
	
	print "---"
	print "Distances: %d" % len(distances)
	count = 0
	for i in distances:
		if i <= 0.35:
			count += 1
	print "Distances less than 0.35: %d" % count
	count = 0
	for i in distances:
		if i <= 0.5:
			count += 1
	print "Distances less than 0.50: %d" % count
	count = 0
	for i in distances:
		if i <= 0.75:
			count += 1
	print "Distances less than 0.75: %d" % count
	print "Finished in %d seconds." % (time() - start_time)


def test_lsh_movie(num_bands):
	start_time = time()
	signature = np.load("Files/MovieSignature.npy")
	pairs = locality_sensitive_hashing_movie(signature[0:2000], num_bands)
	print "---"
	
	loader = load("Files/NormalizedUtilityMatrixCSC.npz")
	n_utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	distances = []
	for pair in pairs:
		num = len(pair)
		for i in xrange(0, num):
			u = n_utility_csc.getcol(pair[i]).toarray()
			for j in xrange(i + 1, num):
				v = n_utility_csc.getcol(pair[j]).toarray()
				distances.add(cosine(u, v))
	
	print "---"
	print "Distances: %d" % len(distances)
	count = 0
	for i in distances:
		if i <= 0.5:
			count += 1
	print "Distances less than 0.50: %d" % count
	print sorted(distances)


=======
    ranking = estimate_user_rating(user_id, movie_id)
    print "User %d Movie %d. Estimated ranking: %d" % (user_id, movie_id, ranking)


def test_lsh(num_bands):
    start_time = time()
    signature = np.load("Files/UserSignature.npy")
    keys, pairs = locality_sensitive_hashing(signature[0:100, :], num_bands)
    print "---"

    loader = load("Files/NormalizedUtilityMatrixCSR.npz")
    n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

    distances = set()
    for key, pair in zip(keys, pairs):
        print key, pair
        num = len(pair)
        for i in xrange(0, num):
            for j in xrange(i + 1, num):
                u = n_utility_csr.getrow(pair[i]).toarray()
                v = n_utility_csr.getrow(pair[j]).toarray()
                distances.add(cosine(u, v))

    print "---"
    print "Distances: %d" % len(distances)
    count = 0
    for i in distances:
        if i <= 0.5:
            count += 1
    print "Distances less than 0.50: %d" % count
    print sorted(distances)


def test_lsh_movie(num_bands):
    start_time = time()
    signature = np.load("Files/MovieSignature.npy")
    keys, pairs = locality_sensitive_hashing_movie(signature[0:2000], num_bands)
    print "---"

    loader = load("Files/NormalizedUtilityMatrixCSC.npz")
    n_utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

    distances = set()
    for key, pair in zip(keys, pairs):
        print key, pair
        num = len(pair)
        for i in xrange(0, num):
            for j in xrange(i + 1, num):
                u = n_utility_csc.getcol(pair[i]).toarray()
                v = n_utility_csc.getcol(pair[j]).toarray()
                distances.add(cosine(u, v))

    print "---"
    print "Distances: %d" % len(distances)
    count = 0
    for i in distances:
        if i <= 0.5:
            count += 1
    print "Distances less than 0.50: %d" % count
    print sorted(distances)


def rmse(acquired_data, test_data):
    diff = acquired_data - test_data
    out = np.sqrt(diff ** 2 / len(test_data.data))
    return out


>>>>>>> Stashed changes
def test_random_hyperplanes_similarity(i=62500, regenerate=False, vector_count=120):
    start_time = time()

    if regenerate:
        generate_user_signature(vector_count)

    # Load necessary matrices.
    loader = load("Files/TrainingMatrixCSR.npz")
    utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
    signature = np.load("Files/UserSignature.npy")

    for j in [1, 2, 62500]:
        u = utility_csr.getrow(i).toarray()
        v = utility_csr.getrow(j).toarray()
        angle, distance = __calculate_similarity_DO_NOT_USE__(i, j, signature)

        print "User %d and Candidate %d" % (i, j)
        print "Angle and Distance: %d degrees, %f" % (angle, distance)
        print "Original Distance:  %f\n" % cosine(u, v)

    print "\n%f seconds elapsed." % (time() - start_time)


def test_all(user_id, movie_id):
	start_time = time()
	print "Latent Factor"
	test_latent_factor(user_id, movie_id)
	print "Finished in %d seconds.\n" % (time() - start_time)
	
	print Toolkit.estimate_rating(user_id, movie_id)

# Test cases.
<<<<<<< Updated upstream
#test_random_hyperplanes_similarity(regenerate=True)
#test_lsh(4)
#test_lsh_speed(4)
#generate_movie_signature()
#test_lsh_movie(8)
test_all(3, 590)
=======
# test_random_hyperplanes_similarity(regenerate=True)
# test_lsh(4)
# generate_movie_signature()
# test_lsh_movie(8)
# test_latent_factor(777, 731)
# test_latent_factor(777, 2027)
# test_latent_factor(777, 11127)
>>>>>>> Stashed changes
