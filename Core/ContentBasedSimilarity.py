import numpy as np
from scipy.sparse import csr_matrix
from numpy import load


# Find Jaccard similarity between two vectors
def find_jaccard_similarity(target_movie, current_movie):
	# common_count = sum(1 for actor in target_movies if actor in current_movie)
	# union = len(set(target_movie+current_movie))
	common_count = 0
	union = len(target_movie) + len(current_movie)
	for attr in target_movie:
		if attr in current_movie:
			common_count += 1
	similarity = float(common_count) / (union - common_count)
	return similarity


# Don't think we need this part
def get_similar_actor(target_movie, id=14):
	loader = load("../Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_of_movies = movie_actor.shape[0]
	commons = []
	target_actors = target_movie.indices
	for movie in xrange(1, num_of_movies):
		current_movie_actors = movie_actor[movie].indices
		similarity = find_jaccard_similarity(target_actors, current_movie_actors)
		if similarity >= 0.5:
			commons.append((movie, similarity))
	commons.sort(key=(lambda x: x[1]))
	return commons


# Finding common movies
# target_movie is of type CSR
def get_similar_movie_by_content(target_movie, content_csr):
	commons = []
	target_content = target_movie.indices
	for movie in xrange(1, content_csr.shape[0]):
		current_elem = content_csr[movie].indices
		similarity = find_jaccard_similarity(target_content, current_elem)
		if similarity >= 0.5:
			commons.append((movie, similarity))
	# Sort descending
	commons.sort(key=(lambda x: x[1]), reverse=True)
	return commons


# Use the method above for all attributes of the movie and find the most similar k movies
def get_similar_movies(target_movie_id):
	loader = load("../Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	loader = load("../Files/YearBasedMatrixCSR.npz")
	movie_year = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	loader = load("../Files/GenreBasedMatrixCSR.npz")
	movie_genre = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# Not sure about this part?? - Tried to obtain the vector of the movie using the movie_id
	target_movie_actor = movie_actor[target_movie_id]
	target_movie_year = movie_year[target_movie_id]
	target_movie_genre = movie_genre[target_movie_id]
	print movie_genre[target_movie_id]

	# Obtain similar movies based on all three attributes
	common_movies_based_on_actor = get_similar_movie_by_content(target_movie_actor, movie_actor)
	common_movies_based_on_year = get_similar_movie_by_content(target_movie_year, movie_year)
	common_movies_based_on_genre = get_similar_movie_by_content(target_movie_genre, movie_genre)

	# Combine the results of all three attributes
	similar_movies = []
	similar_movies.extend(common_movies_based_on_actor)
	similar_movies.extend(common_movies_based_on_year)
	similar_movies.extend(common_movies_based_on_genre)

	# Find the most similar ones among them
	similar_movies.sort(key=lambda x: x[1], reverse=True)

	# Returns the most similar 5 movies based on the content
	return similar_movies[0: 5]

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
print get_similar_movies(32)
