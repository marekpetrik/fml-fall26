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
print("Columns:", list(df.columns))                             # check the exact column names

us = df[df["country"] == "United States"].dropna(subset=["new_cases_smoothed"])
y = us["new_cases_smoothed"].to_numpy()[:400]

plt.plot(y)
plt.xlabel("time")
plt.ylabel("COVID cases")
plt.show()

### --- Construct features

n = len(y)
A = np.column_stack([np.ones(n), np.arange(1, n + 1)])

def f(x):
    return 1/2 * jnp.linalg.norm(A @ x - y)**2      # 1/2 RSS


# the following block adds polynomial features
if False:
    A = np.column_stack([A, np.arange(1, n + 1)**2])

def ReLU(x):
    return np.maximum(x, 0)

# the following block adds splines
if False:
    t10, t11 = -100, 1
    t20, t21 = -200, 1
    t30, t31 = -300, 1
    t = np.arange(1, n + 1)
    A = np.column_stack([A, ReLU(t11 * t + t10)])
    A = np.column_stack([A, ReLU(t21 * t + t20)])
    A = np.column_stack([A, ReLU(t31 * t + t30)])


### --- Linear regression: normal equations

xhat = np.linalg.solve(A.T @ A, A.T @ y)
print(f"Normal equations: x̂ = {xhat}, f(x̂) = {f(xhat)}")

plt.plot(y, label="data", color="black")
plt.plot(A @ xhat, label="ŷ", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases")
plt.legend()
plt.show()

### --- Linear regression: gradient descent (manual)

# returns the iterates x₀, x₁, …, x_K
def gradient_descent(grad_f, x0, alpha, K):
    x = x0
    xs = [x]
    for k in range(K):
        x = x - alpha * grad_f(x)
        xs.append(x)
    return xs

def grad_f1(x):
    return A.T @ (A @ x - y)        # analytical gradient

# hand-selected stepsize
alpha = 0.00000001

# optimal step size (for the quadratic function)
#lams = np.linalg.eigvalsh(2 * A.T @ A)
#alpha = 2 / (max(lams) + min(lams))

K = 100000                          # number of iterations
x0 = np.array([100000.0, -1000.0])
#x0 = np.array([0.0, 0.0])

xs1 = gradient_descent(grad_f1, x0, alpha, K)
print(f"GD (manual): x = {xs1[-1]}, f(x) = {f(xs1[-1])}")

plt.plot(y, label="data", color="black")
plt.plot(A @ xs1[0], label="ŷ₀", color="red", linewidth=2, linestyle="--")
plt.plot(A @ xs1[len(xs1) // 2], label="ŷ₁₀", color="red", linewidth=2, linestyle="--")
plt.plot(A @ xs1[-1], label="ŷ", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases")
plt.legend()
plt.show()

### --- plot convergence

errors = [np.linalg.norm(x - xhat) for x in xs1]
plt.plot(errors)
plt.show()

#max(errors[i+1] / errors[i] for i in range(len(errors) - 1))

### --- Linear regression: gradient descent (autograd)

grad_f2_jit = jax.jit(jax.grad(f))   # compile once: K calls are otherwise slow

def grad_f2(x):
    return np.asarray(grad_f2_jit(x))

xs2 = gradient_descent(grad_f2, x0, alpha, K)
print(f"GD (autograd): x = {xs2[-1]}, f(x) = {f(xs2[-1])}")

plt.plot(y, label="data", color="black")
plt.plot(A @ xs2[0], label="ŷ₀", color="red", linewidth=2)
plt.plot(A @ xs2[10], label="ŷ₁₀", color="red", linewidth=2)
plt.plot(A @ xs2[-1], label="ŷ", color="red", linewidth=2)
plt.xlabel("time")
plt.ylabel("COVID cases")
plt.legend()
plt.show()
