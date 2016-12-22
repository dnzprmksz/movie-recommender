from scipy.sparse import csr_matrix
from numpy import load


# Find Jaccard similarity between two vectors
def find_jaccard_similarity(target_movie, current_movie):
	# common_count = sum(1 for actor in target_movies if actor in current_movie)
	# union = len(set(target_movie+current_movie))
	common_count = 0
	union = len(target_movie) + len(current_movie)
	if union == 0:
		return 0
	for attr in target_movie:
		if attr in current_movie:
			common_count += 1
	if union == common_count:
		return 1
	similarity = float(common_count) / (union - common_count)
	return similarity


# Finding common movies
# target_movie is of type CSR
def __get_similar_movie_by_content__(target_movie, content_csr):
	commons = []
	target_content = target_movie.indices
	for movie in xrange(1, content_csr.shape[0]):
		current_elem = content_csr[movie].indices
		similarity = find_jaccard_similarity(target_content, current_elem)
		commons.append(similarity)
	return commons


# Use the method above for all attributes of the movie and find the most similar k movies
def get_similar_movies(target_movie_id, movie_count=10, year_weight=0.1, genre_weight=0.4, actor_weight=0.5):
	loader = load("../Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	loader = load("../Files/YearBasedMatrixCSR.npz")
	movie_year = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	loader = load("../Files/GenreBasedMatrixCSR.npz")
	movie_genre = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	target_movie_actor = movie_actor[target_movie_id]
	target_movie_year = movie_year[target_movie_id]
	target_movie_genre = movie_genre[target_movie_id]

	# Obtain similar movies based on all three attributes
	common_movies_actor = __get_similar_movie_by_content__(target_movie_actor, movie_actor)
	common_movies_year = __get_similar_movie_by_content__(target_movie_year, movie_year)
	common_movies_genre = __get_similar_movie_by_content__(target_movie_genre, movie_genre)

	similar_movies = []
	for movie_id, data in enumerate(zip(common_movies_actor, common_movies_year, common_movies_genre), 1):
		actor, year, genre = data
		# print actor, year, genre
		similar_movies.append((movie_id, actor*actor_weight + year*year_weight + genre*genre_weight))
	
	# Find the most similar ones among them.
	similar_movies.sort(key=lambda x: x[1], reverse=True)
	
	# Returns the most similar 5 movies based on the content
	return similar_movies[1: movie_count]

# =======SAMPLE USAGE======
# loader = load("../Files/ActorBasedMatrixCSR.npz")
# movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# loader = load("../Files/YearBasedMatrixCSR.npz")
# movie_year = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# loader = load("../Files/GenreBasedMatrixCSR.npz")
# movie_genre = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# temp_actor = csr_matrix((1, 94222), dtype=np.int8)
# temp_actor[0, 1] = 1
# temp_actor[0, 2] = 1
# temp_actor[0, 3] = 1
# temp_actor[0, 4] = 1
# temp_actor[0, 5] = 1
# print get_similar_movie_by_content(temp_actor[0], movie_actor)

# temp_genre = csr_matrix((1, 94222), dtype=np.int8)
# temp_genre[0, 1] = 1
# temp_genre[0, 2] = 1
# temp_genre[0, 3] = 1
# print get_similar_movie_by_content(temp_genre[0], movie_genre)

# temp_year = csr_matrix((1, 94222), dtype=np.int8)
# temp_year[0, 1999] = 1
# print get_similar_movie_by_content(temp_year[0], movie_year)

# Sample Usage:
# # Find the most similar movies for movie_id = 32
# print get_similar_movies(4)
