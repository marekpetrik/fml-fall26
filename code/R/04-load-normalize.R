### Load the matrix
# See: https://rdrr.io/r/utils/read.table.html
#      https://rdrr.io/r/base/matrix.html
F <- read.csv("advertising.csv")
A <- as.matrix(F)


### Compute the dot product between vectors
# https://rdrr.io/r/base/matmult.html  (%*% is the dot product for vectors)
v <- A[, 1]
w <- A[, 2]

print(drop(v %*% w))


### Compute the correlation coefficient
# https://rdrr.io/r/stats/cor.html
print(cor(v, w))

# center features (there is a more elegant way too!)
n <- length(v)
v <- v - drop((rep(1, n) / n) %*% v) * rep(1, n)
w <- w - drop((rep(1, n) / n) %*% w) * rep(1, n)

# check that they are centered
print(sum(v))
print(sum(w))

# compute the correlation coefficient
# https://rdrr.io/r/base/norm.html
print(drop(v %*% w) / norm(v, "2") / norm(w, "2"))


### Normalize the features to sum to one
# https://rdrr.io/r/base/diag.html
A2 <- A %*% diag(drop(1 / (t(A) %*% rep(1, n))))

print(drop(t(A2) %*% rep(1, n)))
