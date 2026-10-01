### Choose vectors and variables

e1 <- c(1, 1)
e1 <- e1 / sqrt(sum(e1^2))
l1 <- 0.2
e2 <- c(1, -1)
e2 <- e2 / sqrt(sum(e2^2))
l2 <- 1
stopifnot(abs(sum(e1 * e2)) <= 1e-5)    # check that they are orthogonal

cat("Start with a sum of squares: \n")
cat(sprintf("%.1f (%.1f x₁ %+.1f x₂)²  + %.1f (%.1f x₁ %+.1f x₂)²\n",
            l1, e1[1], e1[2], l2, e2[1], e2[2]))


### Construct quadratic function

Q <- cbind(e1, e2)
L <- diag(c(l1, l2))
A <- Q %*% L %*% t(Q)

cat("Turn to a quadratic function: \n")
cat(sprintf("%.1f x₁² %+.1f x₁x₂ %+.1f x₂²\n", A[1,1], A[1,2] + A[2,1], A[2,2]))

### Recover the sum of quadratics

# https://rdrr.io/r/base/eigen.html
eig <- eigen(A, symmetric = TRUE)   # eigenvectors are unique only up to sign
l <- eig$values                     # note: R sorts eigenvalues in decreasing order
Q <- eig$vectors

cat("Recover the sum of squares: \n")
cat(sprintf("%.1f (%.1f x₁ %+.1f x₂)²  + %.1f (%.1f x₁ %+.1f x₂)²\n",
            l[1], Q[1,1], Q[2,1], l[2], Q[1,2], Q[2,2]))

### Plot the quadratic function

x1 <- seq(-2, 2, length.out = 100)
x2 <- seq(-2, 2, length.out = 100)
f <- function(x1, x2) A[1,1] * x1^2 + (A[1,2] + A[2,1]) * x1 * x2 + A[2,2] * x2^2
z <- outer(x1, x2, f)   # z[i,j] = f(x1[i], x2[j])

# https://rdrr.io/r/graphics/contour.html
contour(x1, x2, z, xlab = expression(x[1]), ylab = expression(x[2]), asp = 1)

# https://rdrr.io/r/graphics/arrows.html
arrows(0, 0, l1 * e1[1], l1 * e1[2], col = "red", lwd = 2, length = 0.1)
arrows(0, 0, l2 * e2[1], l2 * e2[2], col = "blue", lwd = 2, length = 0.1)
legend("topright", legend = expression(lambda[1] %.% e[1], lambda[2] %.% e[2]),
       col = c("red", "blue"), lwd = 2)
