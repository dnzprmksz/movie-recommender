import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
import mysql.connector
import time

start_time = time.time()

def create_director_content_matrices(conn):
	cursor = conn.cursor()
	movie_cursor = conn.cursor()

	# Get movies
	movie_cursor.execute("SELECT * FROM movie_list;")
	movies = movie_cursor.fetchall()
	num_of_movies = len(movies)

	# Get number of directors
	director_cursor = conn.cursor()
	director_cursor.execute("SELECT COUNT(*) FROM director_list;")
	num_of_directors = director_cursor.fetchall()[0][0]

	row_director = []
	column_director = []
	data_director = []

	# Create director data
	for idx, movie in enumerate(movies, 1):
		if not movie[r] == 0:
			row_genre.append(movie)
			column_genre.append(movie[4])
			data_genre.append(1)

	director_based_csr = csr_matrix((data_director, (row_director, column_director)), shape=(num_of_movies + 1, len(directors) + 1));
	director_based_csc = director_based_csr.tocsc();
	np.savez("../Files/DirectorBasedMatrixCSR", data=genre_based_csr, indices=genre_based_csr.indices,
	indptr=genre_based_csr.indptr, shape=genre_based_csr.shape)

def create_genre_content_matrices():
	# Connect to mysql database
	conn = mysql.connector.connect(user='root', password='root', port=8889, database='webscale')

	print "Connected to the database successfully."
	cursor = conn.cursor()
	movie_cursor = conn.cursor()
	genre_cursor = conn.cursor()

	# Get movies
	movie_cursor.execute("SELECT * FROM movie_list;")
	movies = movie_cursor.fetchall()
	num_of_movies = len(movies)

	# Get number of genres
	genre_cursor.execute("SELECT COUNT(*) FROM genre_list;")
	num_of_genres = genre_cursor.fetchall()[0][0]

	row_genre = []
	column_genre = []
	data_genre = []


	    # Create genre data
	for idx, movie in enumerate(movies, 1):
		if (str(movie[5])) != '0':
			for genre in str(movie[5]).split('|'):
				row_genre.append(idx)
				column_genre.append(int(genre))
				data_genre.append(1)

	genre_based_csr = csr_matrix((data_genre, (row_genre, column_genre)), shape=(num_of_movies + 1, num_of_genres + 1));
	genre_based_csc = genre_based_csr.tocsc();

	np.savez("../Files/GenreBasedMatrixCSR", data=genre_based_csr.data, indices=genre_based_csr.indices,
		 indptr=genre_based_csr.indptr, shape=genre_based_csr.shape)
	np.savez("../Files/GenreBasedMatrixCSC", data=genre_based_csc.data, indices=genre_based_csc.indices,
		 indptr=genre_based_csc.indptr, shape=genre_based_csc.shape)

def create_year_content_matrices(conn):
	cursor = conn.cursor()
	movie_cursor = conn.cursor()
	actor_cursor = conn.cursor()
	year_cursor = conn.cursor()

	# Get movies
	movie_cursor.execute("SELECT * FROM movie_list;")
	movies = movie_cursor.fetchall()
	num_of_movies = len(movies)


	# Calculate the number of release years
	cursor.execute("SELECT MAX(release_year) FROM movie_list;")
	query = cursor.fetchone()
	max_year = query[0]
	# print max_year
	row_year = []
	column_year = []
	data_year = []

	# Create year data
	for idx, movie in enumerate(movies, 1):
		year = int(movie[3])
		if year == 0:
		    continue
		if year < 1900:
			year = 1800
		elif year < 1950:
			year = 1900
		else:
		    year = year / 10 * 10
		row_year.append(idx)
		column_year.append(year)
		data_year.append(1)

	# Save utility matrices.
	year_based_csr = csr_matrix((data_year, (row_year, column_year)), shape=(num_of_movies + 1, max_year + 1));
	year_based_csc = year_based_csr.tocsc();
	np.savez("../Files/YearBasedMatrixCSR", data=year_based_csr.data, indices=year_based_csr.indices,
		  indptr=year_based_csr.indptr, shape=year_based_csr.shape)
	np.savez("../Files/YearBasedMatrixCSC", data=year_based_csc.data, indices=year_based_csc.indices,
		  indptr=year_based_csc.indptr, shape=year_based_csc.shape)

def create_actor_content_matrices():
	# Connect to mysql database
	conn = mysql.connector.connect(user='root', password='root', port=8889, database='webscale')

	cursor = conn.cursor()
	movie_cursor = conn.cursor()
	actor_cursor = conn.cursor()

	# Get movies
	movie_cursor.execute("SELECT * FROM movie_list;")
	movies = movie_cursor.fetchall()
	num_of_movies = len(movies)

	# Get number of actors
	actor_cursor.execute("SELECT COUNT(*) FROM actor_list;")
	num_of_actors = actor_cursor.fetchone()[0]

	# Generate the utility matrices in CSR format.
	row_actor = []
	column_actor = []
	data_actor = []

	for idx, movie in enumerate(movies, 1):
		if (str(movie[6])) != '0':
			for actor in str(movie[6]).split('|'):
				row_actor.append(idx)
				column_actor.append(int(actor))
				data_actor.append(1)
	actor_based_csr = csr_matrix((data_actor, (row_actor, column_actor)), shape=(num_of_movies + 1, num_of_actors + 1));
	actor_based_csc = actor_based_csr.tocsc();
	np.savez("../Files/ActorBasedMatrixCSR", data=actor_based_csr.data, indices=actor_based_csr.indices,
		 indptr=actor_based_csr.indptr, shape=actor_based_csr.shape)

	np.savez("../Files/ActorBasedMatrixCSC", data=actor_based_csc.data, indices=actor_based_csc.indices,
		 indptr=actor_based_csc.indptr, shape=actor_based_csc.shape)

def create_item_content_matrices():
	conn = mysql.connector.connect(user='root', database='web_scale')
	# create_actor_content_matrices(conn)
	# create_genre_content_matrices(conn)
	create_year_content_matrices(conn)
	# create_director_content_matrices(conn)
	print "%f seconds to finish." % (time.time() - start_time)

create_item_content_matrices()
