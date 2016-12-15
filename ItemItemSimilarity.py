from numpy import load
from scipy.sparse import csc_matrix, csr_matrix
from time import time
from scipy.spatial.distance import cosine
from Toolkit import baseline_estimate


def estimate_by_item_similarity(user_id, movie_id, threshold=0.5):
	start_time = time()
	
	# Load utility matrix as Row anc Column matrices.
	loader = load("Files/TrainingMatrixCSC.npz")
	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/TrainingMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Get the data of target movie and target user.
	target_movie = utility_csc.getcol(movie_id)
	target_user = utility_csr.getrow(user_id)
	movies_of_user = target_user.indices
	num_user_movies = len(movies_of_user)
	
	candidate_movies = []
	similar_movies = []
	upper_term = 0
	lower_term = 0
	
	# Find the movies that are rated by target user.
	for i in movies_of_user:
		candidate_movie = utility_csc.getcol(i)
		distance = cosine(candidate_movie.toarray(), target_movie.toarray())
		candidate_movies.append((i, distance))  # Append candidate movie's id and its cos distance to the list.
	
	# Select the most similar ones.
	candidate_movies.sort(key=(lambda x: x[1]))
	for i in xrange(0, num_user_movies):
		candidate = candidate_movies[i]
		distance = candidate[1]
		if distance <= threshold:
			similar_movies.append(candidate)
	
	# Calculate the weighted average of deviations.
	for i in xrange(0, len(similar_movies)):
		movie_tuple = similar_movies[i]  # Consist of movie id and it's cosine distance.
		m_id = movie_tuple[0]
		m_distance = movie_tuple[1]
		m_rating = utility_csr[user_id, m_id]
		
		m_baseline = baseline_estimate(user_id, m_id)
		deviation = m_rating - m_baseline
		similarity = 1 - m_distance
		
		upper_term += similarity * deviation
		lower_term += similarity
	
	# Calculate the rating.
	rating = baseline_estimate(user_id, movie_id) + (upper_term/lower_term)
	return rating
