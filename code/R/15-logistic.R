# R AD backend: torch (install.packages("torch"); then torch::install_torch())
# https://torch.mlverse.org/docs/articles/using-autograd

library(torch)

### --- Download dataset

url  <- "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
path <- "owid_covid.csv"                           # saved in the current directory

if (file.exists(path)) {
  cat("Using cached data in", path, "\n")
} else {
  cat("Fetching data from", url, "...\n")
  download.file(url, path)
}

df <- read.csv(path)
cat("Data dimensions:", dim(df), "\n")

us <- df[df$country == "United States" & !is.na(df$new_cases_smoothed), ]
cases <- us$new_cases_smoothed[1:400]

### --- Binary labels: are the cases above the median?

threshold <- median(cases)
y <- as.numeric(cases > threshold)  # 1 = high, 0 = low

plot(cases, type = "l", xlab = "time", ylab = "COVID cases")
abline(h = threshold, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = c("data", "median"), col = c("black", "red"), lty = c(1, 2), lwd = c(1, 2))

if (interactive()) readline("\npress enter to continue")

### --- Logistic regression: Newton's method (NO NEED TO KNOW)

n <- length(y)
t <- (1:n) / n                   # rescale time to (0,1] to make gradient descent converge faster
A <- cbind(1, t)                 # features: intercept and time

sigma <- function(z) 1 / (1 + exp(-z))   # logistic function: P[yᵢ = 1] = σ(aᵢ'x)

f <- function(x) 1/n * sum(log1p(exp(A %*% x)) - y * (A %*% x))  # (mean) negative log-likelihood

grad_f1 <- function(x) as.vector(1/n * t(A) %*% (sigma(A %*% x) - y))   # analytical gradient
hess_f <- function(x) {
  s <- as.vector(sigma(A %*% x))
  1/n * t(A) %*% diag(s * (1 - s)) %*% A                               # Hessian
}

# no closed form like the normal equations for linear regression;
# each Newton step solves weighted normal equations
newton <- function(grad_f, hess_f, x0, K) {
  x <- x0
  for (k in 1:K) {
    x <- x - solve(hess_f(x), grad_f(x))
  }
  x
}

x0 <- c(0, 0)
xhat <- newton(grad_f1, hess_f, x0, 10)
cat("Newton's method: x̂ =", xhat, ", f(x̂) =", f(xhat), "\n")

plot(y, type = "l", xlab = "time", ylab = "cases > median")
lines(sigma(A %*% xhat), col = "red", lwd = 2)
legend("topleft", legend = c("data", "logistic regression"), col = c("black", "red"), lwd = c(1, 2))

if (interactive()) readline("\npress enter to continue")

### --- Logistic regression: gradient descent (manual)

# returns the iterates x₀, x₁, …, x_K (as a list)
gradient_descent <- function(grad_f, x0, alpha, K) {
  x <- x0
  xs <- list(x)
  for (k in 1:K) {
    x <- x - alpha * grad_f(x)
    xs[[k + 1]] <- x
  }
  xs
}

L <- norm(t(A) %*% A, "2") / (4 * n)   # Lipschitz constant of ∇f (since σ'(z) ≤ 1/4)
alpha <- 1 / L                         # step size
K <- 100                               # number of iterations

xs1 <- gradient_descent(grad_f1, x0, alpha, K)
cat("GD (manual): x =", xs1[[K + 1]], ", f(x) =", f(xs1[[K + 1]]), "\n")

### --- Logistic regression: gradient descent (autograd)

# the AD library needs the function written with its own tensor operations
f_torch <- function(x) {
  At <- torch_tensor(A, dtype = torch_float64())
  yt <- torch_tensor(y, dtype = torch_float64())
  z <- torch_matmul(At, x)
  1/n * torch_sum(torch_log1p(torch_exp(z)) - yt * z)
}

grad_f2 <- function(x) {
  xt <- torch_tensor(x, dtype = torch_float64(), requires_grad = TRUE)
  f_torch(xt)$backward()            # reverse mode: fills in xt$grad
  as_array(xt$grad)
}

xs2 <- gradient_descent(grad_f2, x0, alpha, K)
cat("GD (autograd): x =", xs2[[K + 1]], ", f(x) =", f(xs2[[K + 1]]), "\n")

# plot the evolution of the objective (gap to the optimum from Newton's method)
plot(0:K, sapply(xs1, f) - f(xhat), type = "l", log = "y", col = "blue", lwd = 2,
     xlab = "iteration", ylab = expression(f(x) - f(hat(x))))
lines(0:K, sapply(xs2, f) - f(xhat), col = "green", lty = 2, lwd = 2)
legend("topright", legend = c("GD (manual)", "GD (autograd)"), col = c("blue", "green"),
       lty = c(1, 2), lwd = 2)
