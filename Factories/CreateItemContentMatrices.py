import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time

start_time = time.time()

# Connect to mysql database
conn = mysql.connector.connect(user='root', password='root', port=8889, database='webscale')
print "Connected to the database successfully."
cursor = conn.cursor()
movie_cursor = conn.cursor()
actor_cursor = conn.cursor()
director_cursor = conn.cursor()
genre_cursor = conn.cursor()
year_cursor = conn.cursor()

# Get movies
movie_cursor.execute("SELECT * FROM movie_list;")
movies = movie_cursor.fetchall()
num_of_movies = len(movies)

# Get number of actors
actor_cursor.execute("SELECT COUNT(*) FROM actor_list;")
num_of_actors = actor_cursor.fetchone()[0]

# Get number of directors
director_cursor.execute("SELECT COUNT(*) FROM director_list;")
num_of_directors = director_cursor.fetchall()[0][0]

# Get number of genres
genre_cursor.execute("SELECT COUNT(*) FROM genre_list;")
num_of_genres = genre_cursor.fetchall()[0][0]

# Calculate the number of release years
cursor.execute("SELECT release_year FROM movie_list WHERE release_year = (SELECT MAX(release_year) FROM movie_list);")
query = cursor.fetchone()
max_year = query[0]
# print max_year

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

for idx, movie in enumerate(movies, 1):
    if (str(movie[6])) != '0':
        for actor in str(movie[6]).split('|'):
            row_actor.append(idx)
            column_actor.append(int(actor))
            data_actor.append(1)

    # Create genre data
    if (str(movie[5])) != '0':
        for genre in str(movie[5]).split('|'):
            row_genre.append(idx)
            column_genre.append(int(genre))
            data_genre.append(1)

    # # Create director data
    # if not movie[r] == 0:
    #     row_genre.append(movie)
    #     column_genre.append(movie[4])
    #     data_genre.append(1)
    # Create year data
    if int(movie[3]) != 0:
        row_year.append(idx)
        column_year.append(int(movie[3]))
        data_year.append(1)

actor_based_csr = csr_matrix((data_actor, (row_actor, column_actor)), shape=(num_of_movies + 1, num_of_actors + 1));
genre_based_csr = csr_matrix((data_genre, (row_genre, column_genre)), shape=(num_of_movies + 1, num_of_genres + 1));
# director_based_csr = csr_matrix((data_director, (row_director, column_director)), shape=(num_of_movies + 1, len(directors) + 1));
year_based_csr = csr_matrix((data_year, (row_year, column_year)), shape=(num_of_movies + 1, max_year + 1));

actor_based_csc = actor_based_csr.tocsc();
genre_based_csc = genre_based_csr.tocsc();
# director_based_csc = director_based_csr.tocsc();
year_based_csc = year_based_csr.tocsc();

# Save utility matrices.
# np.savez("../Files/DirectorBasedMatrixCSR", data=genre_based_csr, indices=genre_based_csr.indices,
# indptr=genre_based_csr.indptr, shape=genre_based_csr.shape)

print "%f seconds to finish." % (time.time() - start_time)
