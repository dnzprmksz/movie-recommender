import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time

start_time = time.time()

conn = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='webscale')
print "Connected to the database successfully."
cursor = conn.cursor()

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
np.savez("UtilityMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr, shape=utility_csr.shape)
np.savez("UtilityMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr, shape=utility_csc.shape)

conn.close()
print "%f seconds to finish." % (time.time() - start_time)
