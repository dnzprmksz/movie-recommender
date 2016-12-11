from time import time
from numpy import load
from scipy.sparse import csr_matrix, csc_matrix

start_time = time()

loader = load("UtilityMatrixCSR.npz")
utility_csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

loader = load("UtilityMatrixCSC.npz")
utility_csc = csc_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])

print "%f seconds elapsed." % (time() - start_time)
