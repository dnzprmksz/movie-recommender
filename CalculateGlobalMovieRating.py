from time import time
from numpy import load

start_time = time()

loader = load("UtilityMatrixCSR.npz")

sum = 0
count = len(loader["data"])  # Number of total rankings stored as matrix data.

for ranking in loader["data"]:
	sum += ranking

average = sum/count
print "Global movie ranking is %d." % average

# Save the global movie ranking not to calculate every time, it is not changing.
with open('GlobalMovieRating.txt', 'w') as f:
	f.write(str(average))

print "%f seconds elapsed." % (time() - start_time)
