import numpy as np
import time

from scipy.sparse import csc_matrix, csr_matrix
from numpy import load


def create_cb_rating_matrix(content_based_csr_link, normalized_utility_csr_link):
	# Load matrices.
	loader = load(content_based_csr_link)
	content_based_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load(normalized_utility_csr_link)
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	dictionary = {}
	# Trace all users
	for user_id in range(1, utility_csr.shape[0]):
		# Get the rated movies
		movies_rated_by_user = utility_csr[user_id]
		# print "Rated movies:"
		
		# Trace all the rated movies
		for movie_id in movies_rated_by_user.indices:
			# Get the contents of the movie (all actors etc.)
			contents_of_movie = content_based_csr[movie_id]
			# Get the rating for the movie
			rating = movies_rated_by_user[0, movie_id]
			
			# Trace all the contents
			for content_id in contents_of_movie.indices:
				# If content - user has a value in the dictionary, add the new value on it and increase the count
				#  count - will be used for avarage calculation
				if dictionary.has_key((user_id, content_id)):
					dictionary[user_id, content_id][0] += rating
					dictionary[user_id, content_id][1] += 1
					# print "User:", user_id, "to Actor: ", content_id, ": ", dictionary[(user_id, content_id)]
				# If the content - user key does not exist add a new one with count 1
				else:
					dictionary[(user_id, content_id)] = [rating, 1]
					# print "User:", user_id, "to Actor: ", content_id, ": ", dictionary[(user_id, content_id)]
		
		if user_id % 25 == 0:
			print "25 loops passed"
	
	row = []
	col = []
	data = []
							
	for key, value in dictionary.iteritems():
		row.append(key[0])
		col.append(key[1])
		rating = dictionary[key][0] / dictionary[key][1]
		data.append(rating);
		
	content_based_csr = csr_matrix((data, (row, col)), shape=[utility_csr.shape[0], content_based_csr.shape[1]]);
	
	return content_based_csr

start_time = time.time()

actor_based_csr = create_cb_rating_matrix("../Files/ActorBasedMatrixCSR.npz", "Files/NormalizedUtilityMatrixCSR.npz")
np.savez("../Files/ActorBasedCSR", data=actor_based_csr.data, indices=actor_based_csr.indices, indptr=actor_based_csr.indptr, shape=actor_based_csr.shape)
year_based_csr = create_cb_rating_matrix("../Files/YearBasedMatrixCSR.npz", "Files/NormalizedUtilityMatrixCSR.npz")
np.savez("../Files/YearBasedCSR", data=year_based_csr.data, indices=year_based_csr.indices, indptr=year_based_csr.indptr, shape=year_based_csr.shape)
genre_based_csr = create_cb_rating_matrix("../Files/GenreBasedMatrixCSR.npz", "Files/NormalizedUtilityMatrixCSR.npz")
np.savez("../Files/GenreBasedCSR", data=genre_based_csr.data, indices=genre_based_csr.indices, indptr=genre_based_csr.indptr, shape=genre_based_csr.shape)

print "%f seconds to finish." % (time.time() - start_time)



