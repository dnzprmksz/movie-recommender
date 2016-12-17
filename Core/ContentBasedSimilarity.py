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


# Find most similar 5 movies with the given vector for year

# Find most similar 5 movies with the given vector for genre

# Takes 5x1 vectors from all 3 different contents

temp = csr_matrix((1, 94222), dtype=np.int8)
temp[0, 1] = 1
temp[0, 2] = 1
temp[0, 3] = 1
temp[0, 4] = 1
temp[0, 5] = 1

print get_similar_actor(temp.getrow(0))
#print temp.indices[1]

 #print count_ones([1,0,0,1,0,0,0,0,1], [1,0,0,1,0,0,1,1,1])
