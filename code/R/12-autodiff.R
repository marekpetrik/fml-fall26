# An example of using automatic differentiation
# How AutoDiff works: https://arxiv.org/abs/2501.14787
# R AD backend: torch (install.packages("torch"); then torch::install_torch())
# https://torch.mlverse.org/docs/articles/using-autograd

library(torch)

### Experiments with projections

A <- matrix(c(3, 1, 1, 2), 2, 2, byrow = TRUE)
b <- c(1, 2)

xhat <- solve(A, b)
p <- A %*% xhat

### Projections as minimizing a norm (differentiability issues)

# the AD library needs the function written with its own tensor operations
f <- function(x) 1/2 * sum((A %*% x - b)^2)
f_torch <- function(x) {
  At <- torch_tensor(A, dtype = torch_float64())
  bt <- torch_tensor(b, dtype = torch_float64())
  1/2 * torch_norm(torch_matmul(At, x) - bt)^2
}

## Plots the function
x1 <- seq(-5, 5, length.out = 100)
x2 <- seq(-5, 5, length.out = 100)
z <- outer(x1, x2, Vectorize(function(a1, a2) f(c(a1, a2))))   # z[i,j] = f(x1[i], x2[j])

contour(x1, x2, z, xlab = expression(x[1]), ylab = expression(x[2]), asp = 1)
points(xhat[1], xhat[2], pch = 19, col = "blue")   # minimum

# plot the gradient
grad_f1 <- function(x) as.vector(t(A) %*% A %*% x - t(A) %*% b)
grad_f2 <- function(x) {
  xt <- torch_tensor(x, dtype = torch_float64(), requires_grad = TRUE)
  f_torch(xt)$backward()            # reverse mode: fills in xt$grad
  as_array(xt$grad)
}
x <- c(0, 0)
grad_fx <- grad_f1(x)

points(x[1], x[2], pch = 19, col = "red")   # eval point
arrows(x[1], x[2], x[1] + 0.1 * grad_fx[1], x[2] + 0.1 * grad_fx[2],
       col = "red", lwd = 2, length = 0.1)
legend("topright", legend = expression(hat(x), x, 0.1 %.% nabla * f(x)), col = c("blue", "red", "red"),
       pch = c(19, 19, NA), lty = c(NA, NA, 1), lwd = 2)

cat("Analytical:", grad_f1(x), "\n")
cat("torch:", grad_f2(x), "\n")

if (interactive()) readline("\npress enter to continue")

### Experiments with another function

f <- function(x) log(1 + sum(x * x))
f_torch <- function(x) torch_log(1 + torch_dot(x, x))

## Plots the function
z <- outer(x1, x2, Vectorize(function(a1, a2) f(c(a1, a2))))

contour(x1, x2, z, xlab = expression(x[1]), ylab = expression(x[2]), asp = 1)

# plot the gradient
grad_f1 <- function(x) 2 / (1 + sum(x * x)) * x
x <- c(3, 0)
grad_fx <- grad_f1(x)

points(x[1], x[2], pch = 19, col = "red")   # eval point
arrows(x[1], x[2], x[1] + grad_fx[1], x[2] + grad_fx[2],
       col = "red", lwd = 2, length = 0.1)
legend("topright", legend = expression(x, nabla * f(x)), col = "red",
       pch = c(19, NA), lty = c(NA, 1), lwd = 2)

cat("Analytical:", grad_f1(x), "\n")
cat("torch:", grad_f2(x), "\n")
