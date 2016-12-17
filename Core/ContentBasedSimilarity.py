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
def get_similar_actor(target_movie, id=14):
	loader = load("Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_of_movies = movie_actor.shape[0]
	similarities = []
	# np_tgt = np.array(target_movie)
	transpose_target = target_movie.T
	for movie in xrange(1, num_of_movies):
		current_movie = (movie_actor.getrow(movie))
		# np_cur = np.array(current_movie)
		# print np_cur[0:5]
		# count = count_ones(np_cur, np_tgt)
		# print type(transpose_target)
		# print type(current_movie)
		count = current_movie.dot(transpose_target)
		# print type(count)
		# print count.data
		if len(count.data) > 0:
			similarities.append([movie, count.data[0]])
		else:
			similarities.append([movie, 0])
	return similarities


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


# print count_ones([1,0,0,1,0,0,0,0,1], [1,0,0,1,0,0,1,1,1])
