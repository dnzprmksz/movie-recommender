import numpy as np
from time import time
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial.distance import cosine
from RandomHyperplanes import __calculate_similarity_DO_NOT_USE__


def estimate_by_user_similarity(user_id, movie_id, signature, utility_csr, utility_csc, threshold=0.5):
    # Load necessary matrices.
    
    candidates = []
    user_count=259138
    for other_user in xrange(1,user_count):
	_, distance = __calculate_similarity_DO_NOT_USE__(user_id, other_user, signature)

	if distance <= threshold:
	    rating = utility_csr[other_user, movie_id]
	    candidates.append((1-distance, rating))

    upper_term = 0
    lower_term = 0
    print candidates
    if len(candidates) == 0:
	return 0
    
    for similarity, rating in candidates:
	if rating > 0:
	    upper_term += similarity * rating
	    lower_term += similarity

    rating = upper_term / lower_term
    return rating



# # Load training matrices.
# loader = load("Files/TrainingMatrixCSR.npz")
# training_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# loader = load("Files/TrainingMatrixCSC.npz")
# training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# signature = np.load("Files/UserSignature.npy")

# estimation = estimate_by_user_similarity(10, 20, signature, training_csr, training_csc)

# print estimation
