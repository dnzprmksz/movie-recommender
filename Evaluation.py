from scipy.sparse import csc_matrix, csr_matrix
from numpy import load
import numpy as np

import UserUserSimilarity
from Toolkit import baseline_estimate
import ItemItemSimilarity
import LatentFactor


def evaluate():
	# Load latent factor matrices.
	p = np.load("Files/SGD_P_100.npy")
	q = np.load("Files/SGD_Q_100.npy")
	
	# Load test matrices.
	loader = load("Files/TestMatrixCSR.npz")
	test_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/TestMatrixCSC.npz")
	test_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	# Load training matrices.
	loader = load("Files/MiniTrainingMatrixCSR.npz")
	training_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("Files/MiniTrainingMatrixCSC.npz")
	training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	
	user_signature = load("Files/UserSignature.npy")
	
	baseline_sum = 0.0
	latent_sum = 0.0
	item_sum = 0.0
	user_sum = 0.0
	
	# Calculate inner part of the square root of the RMSE.
	for i in xrange(1, test_csr.shape[0] - 1):
		movie_index = 0
		row = test_csr.getrow(i)
		for j in row.indices:
			rating = row.data[movie_index]
			movie_index += 1
			# Latent factor
			latent_factor_estimation = LatentFactor.estimate_user_rating(i, j, p, q)
			latent_sum += (latent_factor_estimation - rating) ** 2
			# Baseline estimate
			baseline_estimation = baseline_estimate(i, j, training_csr, training_csc)
			baseline_sum += (baseline_estimation - rating) ** 2
			# Item-Item
			item_estimation = ItemItemSimilarity.estimate_by_item_similarity(i, j, training_csr, training_csc)
			item_sum += (item_estimation - rating) ** 2
			# User-User
			user_estimation = UserUserSimilarity.estimate_by_user_similarity(i, j, user_signature, test_csc)
			user_sum += (user_estimation - rating) ** 2
		# Status
		if i % 100 == 0:
			print "%d/%d completed." % (i, test_csr.shape[0] - 1)
	
	data_size = len(test_csr.data)
	
	# Refactor range from 0-100 to 0-5 with dividing by 20.
	print "\nLatent Factor Estimation"
	print np.sqrt(latent_sum / data_size) / 20
	
	print "\nBaseline Estimation"
	print np.sqrt(baseline_sum / data_size) / 20
	
	print "\nItem-Item Estimation"
	print np.sqrt(item_sum / data_size) / 20
	
	print "\nUser-User Estimation"
	print np.sqrt(user_sum / data_size) / 20
