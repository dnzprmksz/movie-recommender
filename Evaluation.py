from scipy.sparse import csc_matrix, csr_matrix
from numpy import load
import numpy as np

from Toolkit import baseline_estimate
import ItemItemSimilarity
import LatentFactor


def evaluate():
	# Load latent factor matrices.
	p = np.load("Files/SGD_P_125.npy")
	q = np.load("Files/SGD_Q_125.npy")
	
	# Load test matrices.
	loader = load("Files/TrainingMatrixCSR.npz")
	test = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/TrainingMatrixCSC.npz")
	test_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Load training matrices.
	loader = load("Files/TrainingMatrixCSR.npz")
	training_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/TrainingMatrixCSC.npz")
	training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	baseline_sum = 0.0
	latent_sum = 0.0
	item_sum = 0.0
	
	# Calculate inner part of the square root of the RMSE.
	for i in xrange(1, test.shape[0] - 1):
		movie_index = 0
		row = test.getrow(i)
		for j in row.indices:
			rating = row.data[movie_index]
			movie_index += 1
			# Latent factor
			latent_factor_estimation = LatentFactor.estimate_user_rating(i, j, p, q)
			latent_sum += (latent_factor_estimation - rating) ** 2
			# Baseline estimate
			#baseline_estimation = baseline_estimate(i, j, training_csr, training_csc)
			#baseline_sum += (baseline_estimation - rating) ** 2
			# Item-Item
			#item_estimation = ItemItemSimilarity.estimate_by_item_similarity(i, j, training_csr, training_csc)
			#item_sum += (item_estimation - rating) ** 2
		# Status
		if i % 1000 == 0:
			print "%d/%d completed." % (i, test.shape[0] - 1)
	
	data_size = len(test.data)
	
	# Refactor range from 0-100 to 0-5 with dividing by 20.
	print "\nLatent Factor"
	print np.sqrt(latent_sum / data_size) / 20
	
	print "\nBaseline estimate"
	print np.sqrt(baseline_sum / data_size) / 20
	
	print "\nItem-Item estimate"
	print np.sqrt(item_sum / data_size) / 20
