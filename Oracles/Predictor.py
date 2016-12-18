# Combine multiple rating prediction methods from Core to make better estimations.
from Core import ItemItemSimilarity
from Core import LatentFactor
from Core import UserUserSimilarity
from Core.DataLoader import training_csc, training_csr, user_signature, p, q


def predict_rating(user_id, movie_id, item_cofactor=0.4, user_cofactor=0.25, latent_cofactor=0.35, content_cofactor=0.0):
	# Get estimations from individual algorithms.
	item_similarity_rating = ItemItemSimilarity.estimate_by_item_similarity(user_id, movie_id, training_csr, training_csc)
	user_similarity_rating = UserUserSimilarity.estimate_by_user_similarity(user_id, movie_id, user_signature, training_csc)
	latent_factor_rating = LatentFactor.estimate_user_rating(user_id, movie_id, p, q)
	
	print "Item similarity: %f" % item_similarity_rating
	print "User similarity: %f" % user_similarity_rating
	print "Latent factor:   %f" % latent_factor_rating
	
	# Calculate a final rating using the linear combination of the estimations.
	rating = item_cofactor * item_similarity_rating + user_cofactor * user_similarity_rating + latent_cofactor * latent_factor_rating
	return rating
