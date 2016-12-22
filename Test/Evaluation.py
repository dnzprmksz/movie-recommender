import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from Oracles import Predictor

import numpy as np
from numpy import load
from scipy.sparse import csc_matrix, csr_matrix
#from Core import ItemItemSimilarity
#from Core import UserUserSimilarity


def evaluate():
	# Load latent factor matrices.
	p = np.load("../Files/SGD_P_100.npy")
	q = np.load("../Files/SGD_Q_100.npy")

	# Load test matrices.
	loader = load("../Files/UtilityMatrix100CSR.npz")
	test_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("../Files/UtilityMatrix100CSC.npz")
	test_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	# Load training matrices.
	loader = load("../Files/UtilityMatrix100CSR.npz")
	training_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
	loader = load("../Files/UtilityMatrix100CSC.npz")
	training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

	user_signature = load("../Files/UserSignature.npy")

	baseline_sum = 0.0
	latent_sum = 0.0
	item_sum = 0.0
	user_sum = 0.0
	predictor_sum = 0.0
	
	print "Evaluation started."
	
	# Calculate inner part of the square root of the RMSE.
	for i in xrange(1, test_csr.shape[0]):
		movie_index = 0
		row = test_csr[i]
		for j in row.indices:
			rating = row.data[movie_index]
			movie_index += 1
			# Latent factor
			#latent_factor_estimation = LatentFactor.estimate_user_rating(i, j, p, q)
			#latent_sum += (latent_factor_estimation - rating) ** 2
			# Baseline estimate
			#baseline_estimation = baseline_estimate(i, j, training_csr, training_csc)
			#baseline_sum += (baseline_estimation - rating) ** 2
			# Item-Item
			#item_estimation = ItemItemSimilarity.estimate_by_item_similarity(i, j, training_csr, training_csc)
			#item_sum += (item_estimation - rating) ** 2
			# User-User
			#user_estimation = UserUserSimilarity.estimate_by_user_similarity(i, j, user_signature, training_csc)
			#user_sum += (user_estimation - rating) ** 2
			# Predictor
			predictor_estimation = Predictor.predict_rating(i, j)
			predictor_sum += (predictor_estimation - rating) ** 2
		# Status
		if i % 1 == 0:
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
	
	print "\nPredictor Estimation"
	print np.sqrt(predictor_sum / data_size) / 20
