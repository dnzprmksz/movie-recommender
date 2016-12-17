from RandomHyperplanes import calculate_user_similarity


def estimate_by_user_similarity(user_id, movie_id, signature, utility_csc):
	candidates = []
	non_candidates = []
	
	# Find the users that rated the given movie.
	movie_col = utility_csc.getcol(movie_id)
	rated_users = movie_col.indices
	user_ratings = movie_col.data
	
	# Calculate the similarity of these users and store them with their rating to the given movie.
	for i in xrange(0, len(rated_users)):
		candidate_id = rated_users[i]
		if candidate_id != user_id:
			_, distance = calculate_user_similarity(user_id, candidate_id, signature)
			similarity = 1 - distance
			rating = user_ratings[i]
			# Store the similarity/rating data.
			if similarity >= 0.5:
				candidates.append((similarity, rating))
			else:
				non_candidates.append((similarity, rating))
	
	# If there is not enough close users, add some more distant users.
	num_candidates = len(candidates)
	if num_candidates < 4:
		diff = 4 - num_candidates
		non_candidates.sort(key=(lambda x: x[0]), reverse=True)  # Sort the list with respect to the similarity.
		candidates.extend(non_candidates[0:diff])
	
	upper_term = 0
	lower_term = 0
	# Calculate the upper/lower terms of the rating prediction formula.
	for similarity, rating in candidates:
		upper_term += similarity * rating
		lower_term += similarity

	if lower_term > 0:
		rating = upper_term / lower_term
	else:
		rating = 70  # Global average.
	return rating
