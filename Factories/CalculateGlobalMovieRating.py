from time import time
from numpy import load

def calculate_global_movie_rating():
	loader = load("../Files/TrainingMatrixCSR.npz")
	sum = 0
	count = len(loader["data"])  # Number of total rankings stored as matrix data.
	for ranking in loader["data"]:
		sum += ranking
		average = sum/count
	print "Global movie ranking is %d." % average
	with open('../Files/GlobalMovieRating.txt', 'w') as f:
		f.write(str(average))
	print "%f seconds elapsed." % (time() - start_time)
