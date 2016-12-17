import LatentFactor
import numpy as np
from numpy import load
from scipy.sparse import csr_matrix

from Core import RandomHyperplanes


def recommend_movie(user_id):
	# Load necessary matrices.
	loader = load("Files/NormalizedUtilityMatrixCSR.npz")
	n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	p = np.load("Files/SGD_P_125.npy")
	q = np.load("Files/SGD_Q_125.npy")
	user_signature = np.load("Files/UserSignature.npy")
	
	candidate_list = []
	
	# Find the cosine distance of each user to the given user.
	for candidate_id in xrange(1, len(user_signature)):
		angle, distance = RandomHyperplanes.calculate_user_similarity(user_id, candidate_id, user_signature)
		candidate_list.append((candidate_id, distance))
	
	# Sort them by distance and select the closest ones.
	candidate_list.sort(key=(lambda x: x[1]))
	
	if len(candidate_list) > 4:
		similar_user_list = candidate_list[0:4]
	else:
		similar_user_list = candidate_list
	
	movie_list = set()
	# Find the best movies of the most similar users.
	for i in xrange(0, len(similar_user_list)):
		similar_user = similar_user_list[i]
		similar_user_id = similar_user[0]
		similar_user_row = n_utility_csr.getrow(similar_user_id)
		
		movie_id_list = similar_user_row.indices
		data = similar_user_row.data
		
		# Form the best movie list of the similar users.
		for j in xrange(0, len(data)):
			current_movie_id = movie_id_list[j]
			rating = data[j]
			if rating >= 0:  # Check if movie has rated as positive by the user.
				movie_list.add(current_movie_id)
	
	recommendation_list = []
	# Predict the ratings of these movies for the given user.
	for movie_id in movie_list:
		prediction = LatentFactor.estimate_user_rating(user_id, movie_id, p, q)
		recommendation_list.append((movie_id, prediction))  # Append movie to list as (movie_id, rating)
	
	recommendation_list.sort(key=(lambda x: x[1]), reverse=True)
	# Remove the movies that the user already rated.
	user_movie_id_list = n_utility_csr.getrow(user_id).indices
	for movie_id in user_movie_id_list:
		for m_tuple in recommendation_list:
			if m_tuple[0] == movie_id:
				recommendation_list.remove(m_tuple)
	
	return recommendation_list
