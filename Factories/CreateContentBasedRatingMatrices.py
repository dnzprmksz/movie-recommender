import bisect
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
	#utility_csr.shape[0]
	for user_id in range(5, 8):
		movies_rated_by_user = utility_csr[user_id]
		print "Rated movies:", movies_rated_by_user
		
		for movie_id in movies_rated_by_user.indices:
			actors_of_movie = content_based_csr[movie_id]
			rating = movies_rated_by_user[0, movie_id]
			print "Actors:", actors_of_movie.indices, "in movie:", movie_id, "rated as: ", rating
			
			for actor_id in actors_of_movie.indices:
				if dictionary.has_key((user_id, actor_id)):
					dictionary[user_id, actor_id][0] += rating
					dictionary[user_id, actor_id][1] += 1
					print "User:", user_id, "to Actor: ", actor_id, ": ", dictionary[(user_id,actor_id)]

				else:
					dictionary[(user_id, actor_id)] = [rating, 1]
					print "User:", user_id, "to Actor: ", actor_id, ": ",dictionary[(user_id,actor_id)]
	
	return dictionary


# def create_cb_rating_matrix(content_based_csc_link, normalized_utility_csc_link):
# 	# Load matrices.
# 	loader = load(content_based_csc_link)
# 	content_based_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
# 	loader = load(normalized_utility_csc_link)
# 	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
#
# 	# For the result matrix:
# 	total_row = []
# 	total_col = []
# 	total_data = []
#
# 	# Trace all indices of the content type(all actor indices etc.)content_based_csc.shape[1]
# 	for cb_index in range(1, 2):
# 		# Get the movies with indexed content(all comedy movies etc.)
# 		movies_of_content = content_based_csc.getcol(cb_index)
#
# 		row = []
# 		col = []
# 		data = []
# 		count = []
#
# 		# For each movie that includes that content:
# 		for movie_index in movies_of_content.indices:
# 			# Get the ratings of users => (user_index, 0) data
# 			ratings = utility_csc.getcol(movie_index)
#
# 			# Trace ratings, for each user with a rating different than zero:
# 			for index in range(0, len(ratings.indices)):
# 				rating = ratings.data[index]
#
# 				if not rating == 0:
# 					user = ratings.indices[index]
#
# 					# If the user has already a rating on that content (for another movie)
# 					if user in row:
# 						csr_attr_index = row.index(user)
# 						rating = rating*count[csr_attr_index]  # To get the last sum
# 						count[csr_attr_index] += 1
# 						# Get the average of new data and the old one
# 						rating = float(rating + data[csr_attr_index]) / count[csr_attr_index]
# 						data[csr_attr_index] = rating
#
# 					# Insert the user into list, place the data and content info into related list with same index
# 					else:
# 						bisect.insort(row, user)
# 						csr_attr_index = row.index(user)
# 						data.insert(csr_attr_index, rating)
# 						col.insert(csr_attr_index, cb_index)
# 						count.insert(csr_attr_index, 1)
#
# 		total_row += row
# 		total_col += col
# 		total_data += data
#
# 	content_based_csr = csr_matrix((total_data, (total_row, total_col)), shape=[utility_csc.shape[0], content_based_csc.shape[1]]);
#
# 	return content_based_csr


start_time = time.time()
actor_based_csr = create_cb_rating_matrix("Files/ActorBasedMatrixCSR.npz", "Files/NormalizedUtilityMatrixCSR.npz")
print "%f seconds to finish." % (time.time() - start_time)

#start_time = time.time()
#genre_based_csr = create_cb_rating_matrix("Files/GenreBasedMatrixCSC.npz", "Files/NormalizedUtilityMatrixCSC.npz")
#print "%f seconds to finish." % (time.time() - start_time)
