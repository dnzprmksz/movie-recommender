import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time
import random
from numpy import load

start_time = time.time()

def create_test_data():
        conn = mysql.connector.connect(user='root', database='web_scale')
        print "Connected to the database successfully."
        cursor = conn.cursor()
        test_cursor = conn.cursor()

        # Constitute mini training data and test data

        # MINI TRAINING

        # Get the number of movies and users to decide on the shape of the utility matrix.
        cursor.execute("SELECT COUNT(*) FROM user_list WHERE user_id<5000;")
        item = cursor.fetchone()
        num_of_users = item[0]

        cursor.execute("SELECT COUNT(*) FROM movie_list;")
        item = cursor.fetchone()
        num_of_movies = item[0]

        # Get all user/movie ratings from database.
        cursor.execute("SELECT user_id, movie_id, rating FROM ml_rating WHERE user_id < 5000 ORDER BY user_id ASC, movie_id ASC")

        # Declare empty arrays to store user, movie and rating data. They will be converted to CSR and CSC matrices.
        row = []
        col = []
        data = []

        # Fill the vectors corresponding to the users, movies and ratings.
        for query_row in cursor:
                row.append(query_row[0])  # user_id
                col.append(query_row[1])  # movie_id
                data.append(query_row[2])  # rating

        print "%f seconds till loop ends." % (time.time() - start_time)
        utility_csr = csr_matrix((data, (row, col)), shape=(num_of_users + 1, num_of_movies + 1))

        # Generate the utility matrix in CSR format for user-user comparison.
        test_row = []
        test_col = []
        test_data = []
        for i in xrange(1, num_of_users):
        	if i % 1000 == 0:
                    print i
        	if len(utility_csr[i].indices) != 1:
            		victim = random.choice(utility_csr[i].indices)
                	test_row.append(i)
                	test_col.append(victim)
                	test_data.append(utility_csr[i, victim])
                	utility_csr[i, victim] = 0


        utility_csr.eliminate_zeros()
        utility_csc = utility_csr.tocsc()
        test_csr = csr_matrix((test_data, (test_row, test_col)), shape=utility_csr.shape)
        test_csc = test_csr.tocsc()
        # Save utility matrices.
        np.savez("../Files/NewTrainingMatrixCSR", data=test_csr.data, indices=test_csr.indices, indptr=test_csr.indptr, shape=test_csr.shape)
        np.savez("../Files/NewTrainingMatrixCSC", data=test_csc.data, indices=test_csc.indices, indptr=test_csc.indptr, shape=test_csc.shape)
        np.savez("../Files/NewTestMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr, shape=utility_csc.shape)
        np.savez("../Files/NewTestMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr, shape=utility_csr.shape)
        conn.close()

# loader = load("../Files/NewTrainingMatrixCSR.npz")
# csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
# print csr.shape
# print len(csr.data)

# loader = load("../Files/NewTestMatrixCSR.npz")
# csr = csr_matrix((loader["data"], loader["indices"], loader["indptr"]), shape=loader["shape"])
# print csr.shape
# print len(csr.data)
