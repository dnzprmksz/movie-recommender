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

row_year = []
column_year = []
data_year = []


for i in range(0, num_of_movies):
    # Create actor data
    if len((str(movies[i][7])).split('|')) <= 1:
        if (str(movies[i][7])).split('|')[7] != '0':
            row_actor.append(i)
            column_actor.append(movies[i][7])
            data_actor.append(1)
    else:
        for j in len((str(movies[i][7])).split('|')):
            row_actor.append(i)
            column_actor.append((str(movies[i][7])).split('|')[j])
            data_actor.append(1)

    # Create genre data
    if len((str(movies[i][6])).split('|')) <= 1:
        if (str(movies[i][6])).split('|')[6] != '0':
            row_genre.append(i)
            column_genre.append(movies[i][6])
            data_genre.append(1)
    else:
        for j in len((str(movies[i][6])).split('|')):
            row_genre.append(i)
            column_genre.append((str(movies[i][6])).split('|')[j])
            data_genre.append(1)

    # Create director data
    if not movies[i][5] == 0:
        row_genre.append(i)
        column_genre.append(movies[i][5])
        data_genre.append(1)
        
    # Create year data
    if not movies[i][4] == 0:
        row_year.append(i)
        column_year.append(movies[i][4])
        data_year.append(1)


actor_based_csr = csr_matrix((data_actor, (row_actor, column_actor)), shape=(num_of_movies + 1, len(actors) + 1));
genre_based_csr = csr_matrix((data_genre, (row_genre, column_genre)), shape=(num_of_movies + 1, len(genres) + 1));
director_based_csr = csr_matrix((data_director, (row_director, column_director)),
                                shape=(num_of_movies + 1, len(directors) + 1));
year_based_csr = csr_matrix((data_year, (row_year, column_year)), shape=(num_of_movies + 1, max_year + 1));

# Save utility matrices.
np.savez("Files/ActorBasedMatrixCSR", data=actor_based_csr, indices=actor_based_csr.indices, indptr=actor_based_csr.indptr,
         shape=actor_based_csr.shape)
np.savez("Files/GenreBasedMatrixCSR", data=genre_based_csr, indices=genre_based_csr.indices, indptr=genre_based_csr.indptr,
         shape=genre_based_csr.shape)
np.savez("Files/DirectorBasedMatrixCSR", data=genre_based_csr, indices=genre_based_csr.indices, indptr=genre_based_csr.indptr,
         shape=genre_based_csr.shape)
np.savez("Files/YearBasedMatrixCSR", data=genre_based_csr, indices=genre_based_csr.indices, indptr=genre_based_csr.indptr,
         shape=genre_based_csr.shape)

print "%f seconds to finish." % (time.time() - start_time)
