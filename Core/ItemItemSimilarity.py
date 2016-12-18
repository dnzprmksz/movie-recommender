from scipy.spatial.distance import cosine
from Toolkit import baseline_estimate


def estimate_by_item_similarity(user_id, movie_id, utility_csr, utility_csc, threshold=0.5):
	# Get the data of target movie and target user.
	target_movie = utility_csc[:, movie_id]
	target_user = utility_csr[user_id]
	movies_of_user = target_user.indices

	# Find the movies that are rated by target user.
	candidate_movies = []
	for rated_movie_id in movies_of_user:
		candidate_movie = utility_csc[:, rated_movie_id]
                if len(target_movie.data) == 0:
                        continue
		distance = cosine(candidate_movie.toarray(), target_movie.toarray())
		candidate_movies.append((rated_movie_id, distance))  # Append candidate movie's id and its cos distance to the list.

	# Select the most similar ones.
	similar_movies = []
	different_movies = []
	for candidate in candidate_movies:
		distance = candidate[1]
		if distance <= threshold:
			similar_movies.append(candidate)
		else:
			different_movies.append(candidate)

	# If there is not enough similar movie with threshold, then add some more from distant movies.
	num_similar_movies = len(similar_movies)
	if num_similar_movies < 5:
		diff = 5 - num_similar_movies
		different_movies.sort(key=(lambda x: x[1]))  # Sort with respect to the distance.
		similar_movies.extend(different_movies[0:diff])

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
	rating = baseline_estimate(user_id, movie_id, utility_csr, utility_csc)
	if lower_term != 0:
		rating += (upper_term / lower_term)
	return rating
