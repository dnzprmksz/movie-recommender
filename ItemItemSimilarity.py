import numpy as np
from numpy import load, mean
from scipy.sparse import csc_matrix, csr_matrix
from time import time
from scipy.spatial.distance import cosine
from Toolkit import baseline_estimate


def find_item_similarity(user_id, movie_id):
	start_time = time()
	
	loader = load("Files/TrainingMatrixCSC.npz")
	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/TrainingMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	movie_ratings = utility_csc.getrow(user_id)
	user_ratings = utility_csr.getrow(movie_id)

	for j in xrange(0, len(movie_ratings.indices)):
		mov_id = movie_ratings.indices[j]
		rxj = movie_ratings.data[j]
		bxj = baseline_estimate(user_id, movie_ratings.indices[j])
		diff = rxj - bxj
		cosine_dist = cosine(utility_csr.getrow(mov_id).toarray(), user_ratings.toarray())
		mul = cosine_dist*diff

	for j in xrange(0, len(movie_ratings.indices)):
		cosine_dist2 = cosine(utility_csr.getrow(mov_id).toarray(), user_ratings.toarray())

	rating = baseline_estimate(user_id, movie_id)+(mul/cosine_dist2)
	print rating

	print movie_ratings.indices  # movie ids
	print movie_ratings.data  # movie ratings
	print user_ratings.indices  # user ids
	print user_ratings.data  # rating of that movie
	print "%f seconds elapsed." % (time() - start_time)


find_item_similarity(2, 491)
