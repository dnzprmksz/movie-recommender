from scipy.spatial.distance import cosine
from Toolkit import baseline_estimate


def estimate_by_item_similarity(user_id, movie_id, utility_csr, utility_csc, threshold=0.5):
	# Get the data of target movie and target user.
	target_movie = utility_csc.getcol(movie_id)
	target_user = utility_csr.getrow(user_id)
	movies_of_user = target_user.indices
	
	# Find the movies that are rated by target user.
	candidate_movies = []
	for i in movies_of_user:
		candidate_movie = utility_csc.getcol(i)
		distance = cosine(candidate_movie.toarray(), target_movie.toarray())
		candidate_movies.append((i, distance))  # Append candidate movie's id and its cos distance to the list.
	
	# Select the most similar ones.
	candidate_movies.sort(key=(lambda x: x[1]))
	similar_movies = []
	for candidate in candidate_movies:
		distance = candidate[1]
		if distance <= threshold:
			similar_movies.append(candidate)
	# If there is no similar movie with threshold, then take all.
	if len(similar_movies) == 0:
		similar_movies = candidate_movies
	
	upper_term = 0
	lower_term = 0
	# Calculate the weighted average of deviations.

	for movie_tuple in similar_movies:
		m_id = movie_tuple[0]
		m_distance = movie_tuple[1]
		m_rating = utility_csr[user_id, m_id]
		
		m_baseline = baseline_estimate(user_id, m_id, utility_csr, utility_csc)
		deviation = m_rating - m_baseline
		similarity = 1 - m_distance

		upper_term += similarity * deviation
		lower_term += similarity
	
	# Calculate the rating.
	rating = baseline_estimate(user_id, movie_id, utility_csr, utility_csc) + (upper_term/lower_term)
	return rating