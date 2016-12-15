from scipy.sparse import csr_matrix, csc_matrix
from numpy import load, mean


def read_global_movie_rating():
	with open('Files/GlobalMovieRating.txt', 'r') as f:
		return f.read()


def user_rating_deviation(user_id):
	loader = load("Files/TrainingMatrixCSR.npz")
	utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	ratings = utility_csr.getrow(user_id)
	
	# Calculate and return the deviation of the user's rating from the global movie rating.
	return mean(ratings.data) - float(read_global_movie_rating())


def movie_rating_deviation(movie_id):
	loader = load("Files/TrainingMatrixCSC.npz")
	utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	ratings = utility_csc.getcol(movie_id)
	
	# Calculate and return the deviation of the movie's rating from the global movie rating.
	return mean(ratings.data) - float(read_global_movie_rating())


def baseline_estimate(user_id, movie_id):
	return float(read_global_movie_rating()) + user_rating_deviation(user_id) + movie_rating_deviation(movie_id)
