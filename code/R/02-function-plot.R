# https://rdrr.io/r/graphics/curve.html

f <- function(x) exp(-2*x)
df <- function(x) -2*exp(-2*x)
ddf <- function(x) 4*exp(-2*x)

x0 <- -4

f1 <- function(x) f(x0) + (x - x0) * df(x0)
f2 <- function(x) f(x0) + (x - x0) * df(x0) + 0.5 * (x - x0)^2 * ddf(x0)

d <- 4

xlim <- c(x0 - d/2, x0 + d/2)
# make sure all three curves fit in the plot
x <- seq(xlim[1], xlim[2], length.out = 200)
ylim <- range(f(x), f1(x), f2(x))

frame()

curve(f, xlim = xlim, ylim = ylim, xlab = "x", ylab = "",
      col = "black", lty = "solid", lwd = 2)
curve(f1, add = TRUE, col = "blue", lty = "dashed", lwd = 2)
curve(f2, add = TRUE, col = "green", lty = "dotted", lwd = 2)

# https://rdrr.io/r/graphics/legend.html
# plotmath renders the partial derivative symbols: https://rdrr.io/r/grDevices/plotmath.html
legend("topright", legend = expression(f, partialdiff * f, partialdiff^2 * f),
       col = c("black", "blue", "green"),
       lty = c("solid", "dashed", "dotted"), lwd = 2)

