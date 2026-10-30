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
y <- us$new_cases_smoothed[1:400] / 100000        # in 100k cases

plot(y, type = "l", xlab = "time", ylab = "COVID cases (100k)")

### --- Shallow network: one hidden layer with ReLU activations

# gradient descent needs all parameters on a similar scale: rescale time (and cases above)
n <- length(y)
t <- (1:n) / n                   # time in (0,1]

ReLU <- function(z) pmax(z, 0)

# hidden units: hⱼ(t) = ReLU(θⱼ₁ t + θⱼ₀),  output: ŷ(t) = w₀ + Σⱼ wⱼ hⱼ(t)
# with θⱼ₁ = 1 and θⱼ₀ = -knotⱼ, the hidden units are the spline features

# knots from spline regression
knots <- c(0, 100, 200, 300) / n

m <- length(knots)               # number of hidden units

# all parameters in one vector: x = [w₀; w; θ₀; θ₁]
unpack <- function(x) list(w0 = x[1], w = x[2:(m+1)], th0 = x[(m+2):(2*m+1)], th1 = x[(2*m+2):(3*m+1)])

network <- function(x) {
  p <- unpack(x)
  H <- ReLU(sweep(outer(t, p$th1), 2, p$th0, "+"))   # hidden layer: Hᵢⱼ = hⱼ(tᵢ)
  as.vector(p$w0 + H %*% p$w)                        # output layer
}

f <- function(x) 1/(2*n) * sum((network(x) - y)^2)  # (half) mean squared error

### --- Spline regression: normal equations (fixed knots)

th0 <- -knots    # initialize to knots
th1 <- rep(1, m)
A <- cbind(1, ReLU(sweep(outer(t, th1), 2, th0, "+")))   # features: intercept and splines

what <- solve(t(A) %*% A, t(A) %*% y)
xhat <- c(what, th0, th1)
cat("Spline (normal equations): f(x̂) =", f(xhat), "\n")

plot(y, type = "l", xlab = "time", ylab = "COVID cases (100k)")
lines(network(xhat), col = "red", lwd = 2)
legend("topleft", legend = expression(data, hat(y) ~ (spline)), col = c("black", "red"), lwd = c(1, 2))

if (interactive()) readline("\npress enter to continue")

### --- Shallow network: gradient descent (autograd)

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

# the AD library needs the function written with its own tensor operations
tt <- torch_tensor(t, dtype = torch_float64())
yt <- torch_tensor(y, dtype = torch_float64())

f_torch <- function(x) {
  p <- unpack(x)
  H <- torch_relu(torch_outer(tt, p$th1) + p$th0)    # hidden layer
  yhat <- p$w0 + torch_matmul(H, p$w)                # output layer
  1/(2*n) * torch_sum((yhat - yt)^2)
}

grad_f <- function(x) {
  xt <- torch_tensor(x, dtype = torch_float64(), requires_grad = TRUE)
  f_torch(xt)$backward()            # reverse mode: fills in xt$grad
  as_array(xt$grad)
}

alpha <- 0.0005                  # step size
K <- 10000                       # number of iterations
x0 <- xhat                       # start from the spline; GD also moves the knots


# This code would randomize the knots
# x0[(m+2):(2*m+1)] <- runif(4) * 400

xs <- gradient_descent(grad_f, x0, alpha, K)
cat("GD (autograd): f(x) =", f(xs[[K + 1]]), "\n")

# initialize the weights
p <- unpack(xs[[K + 1]])
cat("Learned knots (days):", round(-p$th0 / p$th1 * n), "\n")

# iterate xₖ is xs[[k+1]]
plot(y, type = "l", xlab = "time", ylab = "COVID cases (100k)")
lines(network(xs[[1]]), col = "blue", lty = 2, lwd = 2)
lines(network(xs[[length(xs) %/% 2]]), col = "green", lty = 2, lwd = 2)
lines(network(xs[[K + 1]]), col = "red", lwd = 2)
legend("topleft", legend = expression(data, hat(y)[0] ~ (spline), hat(y)[100], hat(y)),
       col = c("black", "blue", "green", "red"), lwd = c(1, 2, 2, 2), lty = c(1, 2, 2, 1))

if (interactive()) readline("\npress enter to continue")


### Plot the evolution of the objective
plot(0:K, sapply(xs, f), type = "l", col = "red", lwd = 2, xlab = "iteration", ylab = "f(x)")
abline(h = f(xhat), col = "blue", lty = 2, lwd = 2)
legend("topright", legend = c("GD (autograd)", "spline"), col = c("red", "blue"), lwd = 2, lty = c(1, 2))
