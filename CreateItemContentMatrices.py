import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time

start_time = time.time()

# Connect to mysql database
conn = mysql.connector.connect(user='root', password='nazli1234', host='127.0.0.1', database='webscale')
print "Connected to the database successfully."
cursor = conn.cursor()
test_cursor = conn.cursor()

# Get movies
cursor.execute("SELECT * FROM movie_list;")
movies = cursor.fetchone()
num_of_movies = len(movies)

# Get actors
cursor.execute("SELECT id FROM actor_list;")
actors = cursor.fetchall()

# Get directors
cursor.execute("SELECT id FROM director_list;")
directors = cursor.fetchall()

# Get genres
cursor.execute("SELECT id FROM genre_list;")
genres = cursor.fetchall()

# Calculate the number of release years
cursor.execute("SELECT release_year FROM movie_list WHERE release_year = (SELECT MAX(release_year) FROM movie_list);")
query = cursor.fetchone()
max_year = query[0]
cursor.execute("SELECT release_year FROM movie_list WHERE release_year = (SELECT MIN(release_year) FROM movie_list "
               "WHERE release_year>0);")
query = cursor.fetchone()
min_year = query[0]
num_of_years = max_year - min_year

# Generate the utility matrices in CSR format.
row_actor = []
column_actor = []
data_actor = []

row_genre = []
column_genre = []
data_genre = []

row_director = []
column_director = []
data_director = []


for i in range(0, num_of_movies):
    # Create actor data
    if len((str(movies[i][6])).split('|')) <= 1:
        if (str(movies[i][6])).split('|')[6] != '0':
            row_actor.append(i)
            column_actor.append(movies[i][6])
            data_actor.append(1)
    else:
        #loop
        row_actor.append(i)
        column_actor.append()
        data_actor.append(1)

    # Create genre data
    if len((str(movies[i][5])).split('|')) <= 1:
        if (str(movies[i][5])).split('|')[6] != '0':
            row_genre.append(i)
            column_genre.append(movies[i][5])
            data_genre.append(1)
    else:
        #
        row_genre.append(i)
        column_genre.append(movies[i][5])
        data_genre.append(1)

    # Create director data
    if not movies[i][4] == 0:
        row_genre.append(i)
        column_genre.append(movies[i][5])
        data_genre.append(1)


actor_based_csr = csr_matrix((data_actor, (row_actor, column_actor)), shape=(num_of_movies + 1, len(actors) + 1));
genre_based_csr = csr_matrix((data_genre, (row_genre, column_genre)), shape=(num_of_movies + 1, len(genres) + 1));
director_based_csr = csr_matrix((data_director, (row_director, column_director)),
                                shape=(num_of_movies + 1, len(directors) + 1));

# # Save utility matrices.
# np.savez("Files/TrainingMatrixCSR", data=utility_csr.data, indices=utility_csr.indices, indptr=utility_csr.indptr,
#          shape=utility_csr.shape)
# np.savez("Files/TrainingMatrixCSC", data=utility_csc.data, indices=utility_csc.indices, indptr=utility_csc.indptr,
#          shape=utility_csc.shape)

print "%f seconds to finish." % (time.time() - start_time)
