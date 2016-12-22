from random import randint

from Core import ContentBasedSimilarity
from Core.DataLoader import n_utility_csr
from Core.DataLoader import user_signature
from Core import RandomHyperplanes
from Oracles import Predictor


def recommend_movie(user_id, desired_movie_count=10, content_based=False):
	candidate_list = []
	
	# Find the cosine distance of each user to the given user.
	for candidate_id in xrange(1, len(user_signature)):
		_, distance = RandomHyperplanes.calculate_user_similarity(user_id, candidate_id, user_signature)
		candidate_list.append((candidate_id, distance))
	
	# Sort them by distance and select the closest ones.
	candidate_list.sort(key=(lambda x: x[1]))
	
	i = 0
	recommendation_pool = set()
	user_movie_id_list = n_utility_csr[user_id].indices
	# Find the best movies of the most similar users.
	while i < len(candidate_list) and len(recommendation_pool) < desired_movie_count:
		similar_user = candidate_list[i]
		similar_user_id = similar_user[0]
		similar_user_row = n_utility_csr[similar_user_id]
		
		movie_id_list = similar_user_row.indices
		data = similar_user_row.data
		num_movies = len(data)
		
		# Form the best movie list of the similar users.
		j = 0
		normalized_max = max(data)
		threshold = normalized_max * 0.5  # Get the movies that are in the top 25% portion of the similar user.
		while j < num_movies and len(recommendation_pool) < desired_movie_count:
			current_movie_id = movie_id_list[j]
			rating = data[j]
			if rating >= threshold and current_movie_id not in user_movie_id_list:  # Check if movie has rated as positive by the user.
				recommendation_pool.add(current_movie_id)
			j += 1
		i += 1
	
	# Add content based movies to recommendation pool if wanted.
	if content_based:
		# Get the movies that user liked.
		user_data = n_utility_csr[user_id]
		normalized_max = max(user_data.data)
		threshold = normalized_max * 0.6  # Get the movies that are in the top 20% portion of the target user.
		users_top_movies = user_data.indices[user_data.data >= threshold]
		random_movie_index = randint(0, len(users_top_movies) - 1)  # Range is inclusive on both sides subtract 1.
		random_movie_id = user_data.indices[random_movie_index]
		
		# Find the movies that are similar to the ones that the user liked. Then add them to the recommendation pool.
		additional_movies = ContentBasedSimilarity.get_similar_movies(random_movie_id, 6)
		for additional_movie_id in additional_movies:
			if additional_movie_id not in user_movie_id_list:
				recommendation_pool.add(additional_movie_id)
	
	recommendation_list = []
	# Predict the ratings of these movies for the given user.
	for movie_id in recommendation_pool:
		prediction = Predictor.predict_rating(user_id, movie_id)
		recommendation_list.append((movie_id, prediction))  # Append movie to list as (movie_id, rating)
	
	recommendation_list.sort(key=(lambda x: x[1]), reverse=True)
	return recommendation_list
