import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax        # https://docs.jax.dev/en/latest/automatic-differentiation.html
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # use 64-bit floats like numpy

### --- Download dataset

url  = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
path = "owid_covid.csv"                            # saved in the current directory

if os.path.isfile(path):
    print(f"Using cached data in {path}")
else:
    print(f"Fetching data from {url} ...")
    urllib.request.urlretrieve(url, path)

df = pd.read_csv(path)
print("Data dimensions:", df.shape)

us = df[df["country"] == "United States"].dropna(subset=["new_cases_smoothed"])
y = us["new_cases_smoothed"].to_numpy()[:400] / 100_000      # in 100k cases

plt.plot(y)
plt.xlabel("time")
plt.ylabel("COVID cases (100k)")
plt.show()

### --- Shallow network: one hidden layer with ReLU activations

# gradient descent needs all parameters on a similar scale: rescale time (and cases above)
n = len(y)
t = np.arange(1, n + 1) / n      # time in (0,1]

def ReLU(z):
    return jnp.maximum(z, 0)

# hidden units: hⱼ(t) = ReLU(θⱼ₁ t + θⱼ₀),  output: ŷ(t) = w₀ + Σⱼ wⱼ hⱼ(t)
# with θⱼ₁ = 1 and θⱼ₀ = -knotⱼ, the hidden units are the spline features

# knots from spline regression
knots = np.array([0, 100, 200, 300]) / n

m = len(knots)                   # number of hidden units

# all parameters in one vector: x = [w₀; w; θ₀; θ₁]
def unpack(x):
    return x[0], x[1:m+1], x[m+1:2*m+1], x[2*m+1:3*m+1]

def network(x):
    w0, w, th0, th1 = unpack(x)
    H = ReLU(jnp.outer(t, th1) + th0)   # hidden layer: Hᵢⱼ = hⱼ(tᵢ)
    return w0 + H @ w                   # output layer

@jax.jit
def f(x):
    return 1/(2*n) * jnp.linalg.norm(network(x) - y)**2    # (half) mean squared error

### --- Spline regression: normal equations (fixed knots)

th0 = -knots    # initialize to knots
th1 = np.ones(m)
A = np.column_stack([np.ones(n), ReLU(np.outer(t, th1) + th0)])   # features: intercept and splines

what = np.linalg.solve(A.T @ A, A.T @ y)
xhat = np.concatenate([what, th0, th1])
print(f"Spline (normal equations): f(x̂) = {f(xhat)}")

plt.plot(y, label="data", color="black")
plt.plot(network(xhat), label="ŷ (spline)", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases (100k)")
plt.legend()
plt.show()

### --- Shallow network: gradient descent (autograd)

# returns the iterates x₀, x₁, …, x_K
def gradient_descent(grad_f, x0, alpha, K):
    x = x0
    xs = [x]
    for k in range(K):
        x = x - alpha * grad_f(x)
        xs.append(x)
    return xs

grad_f_jit = jax.jit(jax.grad(f))   # compile once: K calls are otherwise slow

def grad_f(x):
    return np.asarray(grad_f_jit(x))

alpha = 0.0005                   # step size
K = 10_000                       # number of iterations
x0 = xhat                        # start from the spline; GD also moves the knots


# This code would randomize the knots
# x0 = x0.copy(); x0[m+1:2*m+1] = np.random.rand(4) * 400

xs = gradient_descent(grad_f, x0, alpha, K)
print(f"GD (autograd): f(x) = {f(xs[-1])}")

# initialize the weights
_, _, th0, th1 = unpack(xs[-1])
print("Learned knots (days):", np.round(-th0 / th1 * n).astype(int))

# iterate xₖ is xs[k]
plt.plot(y, label="data", color="black")
plt.plot(network(xs[0]), label="ŷ₀ (spline)", color="blue", linestyle="--", linewidth=2)
plt.plot(network(xs[len(xs) // 2]), label="ŷ₁₀₀", color="green", linestyle="--", linewidth=2)
plt.plot(network(xs[-1]), label="ŷ", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases (100k)")
plt.legend()
plt.show()


### Plot the evolution of the objective
plt.plot(range(K + 1), [f(x) for x in xs], label="GD (autograd)", color="red", linewidth=2)
plt.axhline(f(xhat), label="spline", color="blue", linestyle="--", linewidth=2)
plt.xlabel("iteration")
plt.ylabel("f(x)")
plt.legend()
plt.show()
