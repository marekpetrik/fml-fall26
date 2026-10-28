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
cat("Columns:", names(df), "\n")                                # check the exact column names

us <- df[df$country == "United States" & !is.na(df$new_cases_smoothed), ]
y <- us$new_cases_smoothed[1:400]

plot(y, type = "l", xlab = "time", ylab = "COVID cases")

### --- Construct features

n <- length(y)
A <- cbind(1, 1:n)

f <- function(x) 1/2 * sum((A %*% x - y)^2)   # 1/2 RSS


# this adds polynomial features

if (FALSE) {
  A <- cbind(A, (1:n)^2)
}

ReLU <- function(x) pmax(x, 0)

# this adds splines
if (FALSE) {
  t10 <- -100; t11 <- 1
  t20 <- -200; t21 <- 1
  t30 <- -300; t31 <- 1
  A <- cbind(A, ReLU(t11 * (1:n) + t10))
  A <- cbind(A, ReLU(t21 * (1:n) + t20))
  A <- cbind(A, ReLU(t31 * (1:n) + t30))
}


### --- Linear regression: normal equations

xhat <- solve(t(A) %*% A, t(A) %*% y)
cat("Normal equations: x̂ =", xhat, ", f(x̂) =", f(xhat), "\n")

plot(y, type = "l", xlab = "time", ylab = "COVID cases")
lines(A %*% xhat, col = "red", lwd = 2)
legend("topleft", legend = expression(data, hat(y)), col = c("black", "red"), lwd = c(1, 2))

if (interactive()) readline("\npress enter to continue")

### --- Linear regression: gradient descent (manual)

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

grad_f1 <- function(x) as.vector(t(A) %*% (A %*% x - y))   # analytical gradient

alpha <- 0.000000001
K <- 20                          # number of iterations
x0 <- c(100000, 0)

xs1 <- gradient_descent(grad_f1, x0, alpha, K)
cat("GD (manual): x =", xs1[[K + 1]], ", f(x) =", f(xs1[[K + 1]]), "\n")

plot(y, type = "l", xlab = "time", ylab = "COVID cases")
lines(A %*% xs1[[1]], col = "red", lwd = 2)
lines(A %*% xs1[[11]], col = "red", lwd = 2)
lines(A %*% xs1[[K + 1]], col = "red", lwd = 2)
legend("topleft", legend = expression(data, list(hat(y)[0], hat(y)[10], hat(y))), col = c("black", "red"), lwd = c(1, 2))

if (interactive()) readline("\npress enter to continue")

### --- Linear regression: gradient descent (autograd)

# the AD library needs the function written with its own tensor operations
f_torch <- function(x) {
  At <- torch_tensor(A, dtype = torch_float64())
  yt <- torch_tensor(y, dtype = torch_float64())
  1/2 * torch_sum((torch_matmul(At, x) - yt)^2)
}

grad_f2 <- function(x) {
  xt <- torch_tensor(x, dtype = torch_float64(), requires_grad = TRUE)
  f_torch(xt)$backward()            # reverse mode: fills in xt$grad
  as_array(xt$grad)
}

xs2 <- gradient_descent(grad_f2, x0, alpha, K)
cat("GD (autograd): x =", xs2[[K + 1]], ", f(x) =", f(xs2[[K + 1]]), "\n")

plot(y, type = "l", xlab = "time", ylab = "COVID cases")
lines(A %*% xs2[[1]], col = "red", lwd = 2)
lines(A %*% xs2[[11]], col = "red", lwd = 2)
lines(A %*% xs2[[K + 1]], col = "red", lwd = 2)
legend("topleft", legend = expression(data, list(hat(y)[0], hat(y)[10], hat(y))), col = c("black", "red"), lwd = c(1, 2))
