import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time

start_time = time.time()

conn = mysql.connector.connect(user='root', password='nazli1234', host='127.0.0.1', database='webscale')
print "Connected to the database successfully."
cursor = conn.cursor()
test_cursor = conn.cursor()


# Get the number of movies and users to decide on the shape of the utility matrix.
cursor.execute("SELECT COUNT(*) FROM user_list;")
item = cursor.fetchone()
num_of_users = item[0]

cursor.execute("SELECT COUNT(*) FROM movie_list;")
item = cursor.fetchone()
num_of_movies = item[0]

# Get all user/movie ratings from database.
cursor.execute("SELECT user_id, movie_id, rating FROM ml_rating ORDER BY user_id ASC, movie_id ASC")
print "%f seconds till SQL query ends." % (time.time() - start_time)

# Declare empty arrays to store user, movie and rating data. They will be converted to CSR and CSC matrices.
row = []
col = []
data = []

# Fill the vectors corresponding to the users, movies and ratings.
for query_row in cursor:
	row.append(query_row[0])   # user_id
	col.append(query_row[1])   # movie_id
	data.append(query_row[2])  # rating

print "%f seconds till loop ends." % (time.time() - start_time)

# Generate the utility matrix in CSR format for user-user comparison.
utility_csr = csr_matrix((data, (row, col)), shape=(num_of_users + 1, num_of_movies + 1))

# Generate the utility matrix in CSC format for movie-movie comparison.
utility_csc = csc_matrix((data, (row, col)), shape=(num_of_users + 1, num_of_movies + 1))

# Save utility matrices.
np.savez("Files/TrainingMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr, shape=utility_csr.shape)
np.savez("Files/TrainingMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr, shape=utility_csc.shape)

print "%f seconds to finish." % (time.time() - start_time)

start_time2 = time.time()
#create test data
test_cursor.execute("SELECT COUNT(1) FROM user_list WHERE user_id>200000;")
item2 = test_cursor.fetchone()
test_num_of_users = item2[0]

test_cursor.execute("SELECT COUNT(1) FROM movie_list WHERE id>30000;")
item2 = test_cursor.fetchone()
test_num_of_movies = item2[0]

# Get all test user/movie ratings from database.
test_cursor.execute("SELECT user_id, movie_id, rating FROM test_data ORDER BY user_id ASC, movie_id ASC")
print "%f seconds till SQL query ends." % (time.time() - start_time2)

# Declare empty arrays to store user, movie and rating data. They will be converted to CSR and CSC matrices.
test_row = []
test_col = []
test_data = []

# Fill the vectors corresponding to the users, movies and ratings.
for query_row1 in test_cursor:
	test_row.append(query_row1[0]-200000)  # user_id
	test_col.append(query_row1[1]-30000)  # movie_id
	test_data.append(query_row1[2])  # rating

print "%f seconds till loop ends." % (time.time() - start_time2)

# Generate the utility matrix in CSR format for user-user comparison.
test_utility_csr = csr_matrix((test_data, (test_row, test_col)), shape=(test_num_of_users + 1, test_num_of_movies + 1))

# Generate the utility matrix in CSC format for movie-movie comparison.
test_utility_csc = csc_matrix((test_data, (test_row, test_col)), shape=(test_num_of_users + 1, test_num_of_movies + 1))

# Save utility matrices.
np.savez("Files/TestMatrixCSR", data=test_utility_csr.data, indices=test_utility_csr.indices, indptr=test_utility_csr.indptr, shape=test_utility_csr.shape)
np.savez("Files/TestMatrixCSC", data=test_utility_csc.data, indices=test_utility_csc.indices, indptr=test_utility_csc.indptr, shape=test_utility_csc.shape)

print "%f seconds to finish." % (time.time() - start_time2)
conn.close()

