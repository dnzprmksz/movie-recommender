import numpy as np
from numpy import mean


def rmse(acquired_data, test_data):
	summation = 0.0
	diff = acquired_data - test_data
	for i in (1, diff.shape[0] - 1):
		for j in diff[i].indices:
			summation += diff[i, j] ** 2
	print np.sqrt(summation / diff.data)


def read_global_movie_rating():
	with open('Files/GlobalMovieRating.txt', 'r') as f:
		return float(f.read())


def user_rating_deviation(user_id, utility_csr):
	ratings = utility_csr.getrow(user_id)
	# Calculate and return the deviation of the user's rating from the global movie rating.
	if len(ratings.data) > 0:
		return mean(ratings.data) - 70.0
	else:
		return 0.0


def movie_rating_deviation(movie_id, utility_csc):
	ratings = utility_csc.getcol(movie_id)
	# Calculate and return the deviation of the movie's rating from the global movie rating.
	if len(ratings.data) > 0:
		return mean(ratings.data) - 70.0
	else:
		return 0.0


def baseline_estimate(user_id, movie_id, utility_csr, utility_csc):
	return 70.0 + user_rating_deviation(user_id, utility_csr) + movie_rating_deviation(movie_id, utility_csc)
