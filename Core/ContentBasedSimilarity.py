import numpy as np
from scipy.sparse import csr_matrix
from numpy import load


# def count_ones(vector1, vector2):
# 	count = 0
# 	print len(vector1)
# 	for i in xrange(len(vector1)):
# 		if vector1[i] == vector2[i] and vector1[i] == 1:
# 			count += 1
# 	return count

# Find most similar 5 movies with the given vector for actors
# target_movie is type vector
# id is the given movie_id
def find_jaccard_similarity(target_movie, current_movie):
        # common_count = sum(1 for actor in target_movies if actor in current_movie)
        # union = len(set(target_movie+current_movie))
	common_count = 0
	union = len(target_movie) + len(current_movie)
	for actor in target_movie:
                if actor in current_movie:
                        common_count += 1
	similarity = float(common_count)/(union - common_count)
        return similarity


def get_similar_actor(target_movie, id=14):
	loader = load("../Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_of_movies = movie_actor.shape[0]
	similarities = []
	commons = []
	# np_tgt = np.array(target_movie)
	target_actors = target_movie.indices
	for movie in xrange(1, num_of_movies):
                current_movie_actors = movie_actor[movie].indices
		similarity = find_jaccard_similarity(target_actors, current_movie_actors)
                if similarity >= 0.5:
                        commons.append((movie, similarity))
        commons.sort(key=(lambda x:x[1]))
	return commons


def get_similar_movie_by_content(target_movie, content_csr):
	similarities = []
	commons = []
	target_content = target_movie.indices
	for movie in xrange(1, content_csr.shape[0]):
                current_elem = content_csr[movie].indices
		similarity = find_jaccard_similarity(target_content, current_elem)
                if similarity >= 0.5:
                        commons.append((movie, similarity))
        commons.sort(key=(lambda x:x[1]))
	return commons


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
