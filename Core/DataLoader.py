# Import this file to load all matrices, where loading individual matrices results in long and dirty headers.
import os
import sys
import numpy as np

from scipy.sparse import csr_matrix, csc_matrix

sys.path.insert(0, os.path.abspath('..'))

# Utility matrix in row form.
loader = np.load("../Files/TrainingMatrixCSR.npz")
training_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# Utility matrix in column form.
loader = np.load("../Files/TrainingMatrixCSC.npz")
training_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# Normalized utility matrix in row form.
loader = np.load("../Files/NormalizedUtilityMatrixCSR.npz")
n_utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# Normalized utility matrix in column form.
loader = np.load("../Files/NormalizedUtilityMatrixCSC.npz")
n_utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

# Random hyperplanes.
user_signature = np.load("../Files/UserSignature.npy")

# Latent factor.
p = np.load("../Files/SGD_P_100.npy")
q = np.load("../Files/SGD_Q_100.npy")
