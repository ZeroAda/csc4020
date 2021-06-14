import numpy as np
X = np.ones(100,100)
Y = np.ones(100,100)
m =100
permutation = list(np.random.permutation(m))
shuffled_X = X[:, permutation]
shuffled_Y = Y[:, permutation].reshape((1, m))

