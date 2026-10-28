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
cases = us["new_cases_smoothed"].to_numpy()[:400]

### --- Binary labels: are the cases above the median?

threshold = np.median(cases)
y = (cases > threshold).astype(float)  # 1 = high, 0 = low

plt.plot(cases, label="data", color="black")
plt.axhline(threshold, label="median", color="red", linestyle="--", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases")
plt.legend()
plt.show()

### --- Logistic regression: Newton's method (NO NEED TO KNOW)

n = len(y)
t = np.arange(1, n + 1) / n         # rescale time to (0,1] to make gradient descent converge faster
A = np.column_stack([np.ones(n), t])  # features: intercept and time

def sigma(z):
    return 1 / (1 + np.exp(-z))     # logistic function: P[yᵢ = 1] = σ(aᵢ'x)

def f(x):
    return 1/n * jnp.sum(jnp.log1p(jnp.exp(A @ x)) - y * (A @ x))  # (mean) negative log-likelihood

def grad_f1(x):
    return 1/n * A.T @ (sigma(A @ x) - y)                          # analytical gradient

def hess_f(x):
    s = sigma(A @ x)
    return 1/n * A.T @ np.diag(s * (1 - s)) @ A                    # Hessian

# no closed form like the normal equations for linear regression;
# each Newton step solves weighted normal equations
def newton(grad_f, hess_f, x0, K):
    x = x0
    for k in range(K):
        x = x - np.linalg.solve(hess_f(x), grad_f(x))
    return x

x0 = np.zeros(2)
xhat = newton(grad_f1, hess_f, x0, 10)
print(f"Newton's method: x̂ = {xhat}, f(x̂) = {f(xhat)}")

plt.plot(y, label="data", color="black")
plt.plot(sigma(A @ xhat), label="logistic regression", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("cases > median")
plt.legend()
plt.show()

### --- Logistic regression: gradient descent (manual)

# returns the iterates x₀, x₁, …, x_K
def gradient_descent(grad_f, x0, alpha, K):
    x = x0
    xs = [x]
    for k in range(K):
        x = x - alpha * grad_f(x)
        xs.append(x)
    return xs

L = np.linalg.norm(A.T @ A, 2) / (4 * n)  # Lipschitz constant of ∇f (since σ'(z) ≤ 1/4)
alpha = 1 / L                       # step size
K = 100                             # number of iterations

xs1 = gradient_descent(grad_f1, x0, alpha, K)
print(f"GD (manual): x = {xs1[-1]}, f(x) = {f(xs1[-1])}")

### --- Logistic regression: gradient descent (autograd)

def grad_f2(x):
    return np.asarray(jax.grad(f)(x))

xs2 = gradient_descent(grad_f2, x0, alpha, K)
print(f"GD (autograd): x = {xs2[-1]}, f(x) = {f(xs2[-1])}")

# plot the evolution of the objective (gap to the optimum from Newton's method)
plt.semilogy(range(K + 1), [f(x) - f(xhat) for x in xs1], label="GD (manual)", color="blue", linewidth=2)
plt.semilogy(range(K + 1), [f(x) - f(xhat) for x in xs2], label="GD (autograd)",
             color="green", linestyle="--", linewidth=2)
plt.xlabel("iteration")
plt.ylabel("f(x) - f(x̂)")
plt.legend()
plt.show()
