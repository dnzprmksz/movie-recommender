from Core.DataLoader import n_utility_csr
from Core.DataLoader import user_signature
from Core import RandomHyperplanes
from Oracles import Predictor


def recommend_movie(user_id, desired_movie_count=10):
	candidate_list = []
	
	# Find the cosine distance of each user to the given user.
	for candidate_id in xrange(1, len(user_signature)):
		_, distance = RandomHyperplanes.calculate_user_similarity(user_id, candidate_id, user_signature)
		candidate_list.append((candidate_id, distance))
	
	# Sort them by distance and select the closest ones.
	candidate_list.sort(key=(lambda x: x[1]))
	
	i = 0
	movie_list = set()
	user_movie_id_list = n_utility_csr[user_id].indices
	# Find the best movies of the most similar users.
	while i < len(candidate_list) and len(movie_list) < desired_movie_count:
		similar_user = candidate_list[i]
		similar_user_id = similar_user[0]
		similar_user_row = n_utility_csr[similar_user_id]
		
		movie_id_list = similar_user_row.indices
		data = similar_user_row.data
		num_movies = len(data)
		
		# Form the best movie list of the similar users.
		j = 0
		while j < num_movies and len(movie_list) < desired_movie_count:
			current_movie_id = movie_id_list[j]
			rating = data[j]
			if rating >= 0 and current_movie_id not in user_movie_id_list:  # Check if movie has rated as positive by the user.
				movie_list.add(current_movie_id)
			j += 1
		i += 1
	
	recommendation_list = []
	# Predict the ratings of these movies for the given user.
	for movie_id in movie_list:
		prediction = Predictor.predict_rating(user_id, movie_id)
		recommendation_list.append((movie_id, prediction))  # Append movie to list as (movie_id, rating)
	
	recommendation_list.sort(key=(lambda x: x[1]), reverse=True)
	return recommendation_list
