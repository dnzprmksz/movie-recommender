from itertools import izip
import numpy as np
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix

import Toolkit
from RandomHyperplanes import calculate_user_similarity


def estimate_by_user_similarity(user_id, movie_id, signature, utility_csc):
	candidates = []
	non_candidates = []

	# Find the users that rated the given movie.
	movie_col = utility_csc[:, movie_id]
	rated_users = movie_col.indices
	user_ratings = movie_col.data

	# Calculate the similarity of these users and store them with their rating to the given movie.
	for rating, candidate_id in izip(user_ratings, rated_users):
		if candidate_id != user_id:
			_, distance = calculate_user_similarity(user_id, candidate_id, signature)
			similarity = 1 - distance
			# Store the similarity/rating data.
			if similarity >= 0.5:
				candidates.append((similarity, rating))
			else:
				non_candidates.append((similarity, rating))

	# If there is not enough close users, add some more distant users.
	num_candidates = len(candidates)
	if num_candidates < 4:
		diff = 4 - num_candidates
		non_candidates.sort(key=(lambda x: x[0]), reverse=True)  # Sort the list with respect to the similarity.
		candidates.extend(non_candidates[:diff])

	upper_term = 0
	lower_term = 0
	# Calculate the upper/lower terms of the rating prediction formula.
	for similarity, rating in candidates:
		upper_term += similarity * rating
		lower_term += similarity

	# If no one rated this movie return the global average.
	if lower_term != 0:
		rating = upper_term / lower_term
	else:
		rating = Toolkit.read_global_movie_rating()
	return rating


# Load training matrices.
loader = load("../Files/TrainingMatrixCSC.npz")
training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
user_signature = load("../Files/UserSignature.npy")

user_estimation = estimate_by_user_similarity(10, 10, user_signature, training_csc)
print user_estimation
