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
	num_of_actors_in_current_movie = len(current_movie)
	num_of_actors_in_target_movie = len(target_movie)

	print num_of_actors_in_current_movie
	print num_of_actors_in_target_movie

	# If curent movie has no actors, return 0
	if num_of_actors_in_current_movie == 0:
		return 0
	common_count = 0

	union = num_of_actors_in_target_movie + num_of_actors_in_current_movie
	for i in xrange(num_of_actors_in_target_movie):
		for j in xrange(num_of_actors_in_target_movie):
			if target_movie[i] == current_movie[j]:
				common_count += 1
	print "union: ", union
	print "comon count :" ,common_count
	# if no actors are the same, return 0
	if common_count == 0:
		return 0
	#similarity = common_count/(union - common_count)
	return 0


def get_similar_actor(target_movie, id=14):
	loader = load("Files/ActorBasedMatrixCSR.npz")
	movie_actor = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	num_of_movies = movie_actor.shape[0]
	similarities = []
	commons = []
	# np_tgt = np.array(target_movie)
	i = 0;
	target_actors = target_movie.indices
	for movie in xrange(1, num_of_movies):
		current_movie_actors = (movie_actor.getrow(movie)).indices
		find_jaccard_similarity(target_actors, current_movie_actors)
		#commons.append(common_count)

		# np_cur = np.array(current_movie)
		# print np_cur[0:5]
		# count = count_ones(np_cur, np_tgt)
		# print type(transpose_target)
		# print type(current_movie)
		# print type(count)
		# print count.data
		# if len(count.data) > 0:
		# 	similarities.append([movie, count.data[0]])
		# else:
		# 	similarities.append([movie, 0])
	#print movie_actor.getrow(1)
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

get_similar_actor(temp.getrow(0))
#print temp.indices[1]

 #print count_ones([1,0,0,1,0,0,0,0,1], [1,0,0,1,0,0,1,1,1])
