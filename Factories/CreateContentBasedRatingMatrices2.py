import bisect
import time
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix
from numpy import load


def create_cb_rating_matrix(content_based_csc_link, normalized_utility_csc_link):
	# Load matrices.
	loader = load(content_based_csc_link)
	content_based_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load(normalized_utility_csc_link)
	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# For the result matrix:
	total_row = []
	total_col = []
	total_data = []

	# Trace all indices of the content type(all actor indices etc.)
	for cb_index in range(1, content_based_csc.shape[1]):
		# Get the movies with indexed content(all comedy movies etc.)
		movies_of_content = content_based_csc.getcol(cb_index)

		row = []
		col = []
		data = []
		count = []

		# For each movie that includes that content:
		for movie_index in movies_of_content.indices:
			# Get the ratings of users => (user_index, 0) data
			ratings = utility_csc.getcol(movie_index)
			print ratings
			# Trace ratings, for each user with a rating different than zero:0,
			for index in range(0, len(ratings.indices)):
				rating = ratings.data[index]

				if not rating == 0:
					user = ratings.indices[index]

					# If the user has already a rating on that content (for another movie)
					if user in row:
						csr_attr_index = row.index(user)
						rating = rating*count[csr_attr_index]  # To get the last sum
						count[csr_attr_index] += 1
						# Get the average of new data and the old one
						rating = float(rating + data[csr_attr_index]) / count[csr_attr_index]
						data[csr_attr_index] = rating
						print "User:", user, " for content:", cb_index, "changed rating: ", rating
					
					# Insert the user into list, place the data and content info into related list with same index
					else:
						bisect.insort(row, user)
						csr_attr_index = row.index(user)
						data.insert(csr_attr_index, rating)
						col.insert(csr_attr_index, cb_index)
						count.insert(csr_attr_index, 1)
						print "User: ", user, " for content: ", cb_index, "rating: ",rating

		total_row += row
		total_col += col
		total_data += data
		
	# print total_row
	# print total_col
	# print total_data

	content_based_csr = csr_matrix((total_data, (total_row, total_col)), shape=[utility_csc.shape[0], content_based_csc.shape[1]]);

	return content_based_csr


start_time = time.time()
actor_based_csr = create_cb_rating_matrix("../Files/ActorBasedMatrixCSC.npz", "../Files/NormalizedUtilityMatrixCSC.npz")
np.savez("../Files/ActorBasedCSR2", data=actor_based_csr.data, indices=actor_based_csr.indices, indptr=actor_based_csr.indptr, shape=actor_based_csr.shape)
print "%f seconds to finish." % (time.time() - start_time)
