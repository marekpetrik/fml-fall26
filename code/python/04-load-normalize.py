import numpy as np      # https://numpy.org/doc/stable/reference/routines.linalg.html
import pandas as pd     # https://pandas.pydata.org/docs/

### Load the matrix
# See: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
#      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html
F = pd.read_csv("advertising.csv")
A = F.to_numpy()


### Compute the dot product between vectors
# https://numpy.org/doc/stable/reference/generated/numpy.dot.html
v = A[:, 0]
w = A[:, 1]

print(np.dot(v, w))


### Compute the correlation coefficient
# https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
print(np.corrcoef(v, w)[0, 1])

# center features (there is a more elegant way too!)
n = len(v)
v = v - np.dot(np.ones(n)/n, v) * np.ones(n)
w = w - np.dot(np.ones(n)/n, w) * np.ones(n)

# check that they are centered
print(v.sum())
print(w.sum())

# compute the correlation coefficient
# https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
print(np.dot(v, w) / np.linalg.norm(v) / np.linalg.norm(w))


### Normalize the features to sum to one
# https://numpy.org/doc/stable/reference/generated/numpy.diag.html
A2 = A @ np.diag(1 / (A.T @ np.ones(n)))

print(A2.T @ np.ones(n))
