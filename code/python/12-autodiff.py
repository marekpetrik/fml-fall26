# An example of using automatic differentiation
# How AutoDiff works: https://arxiv.org/abs/2501.14787
# For some limitations, see:
# https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html

import numpy as np
import matplotlib.pyplot as plt

import torch      # https://docs.pytorch.org/docs/stable/autograd.html
import jax        # https://docs.jax.dev/en/latest/automatic-differentiation.html
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)   # use 64-bit floats like numpy

### Experiments with projections

A = np.array([[3.0, 1], [1, 2]])
b = np.array([1.0, 2])

xhat = np.linalg.solve(A, b)
p = A @ xhat

### Projections as minimizing a norm (differentiability issues)

# each AD library needs the function written with its own array operations
def f(x):        return 1/2 * np.linalg.norm(A @ x - b)**2
def f_torch(x):  return 1/2 * torch.linalg.norm(torch.tensor(A) @ x - torch.tensor(b))**2
def f_jax(x):    return 1/2 * jnp.linalg.norm(A @ x - b)**2

## Plots the function
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = np.vectorize(lambda a1, a2: f(np.array([a1, a2])))(X1, X2)

plt.figure(figsize=(5, 5))
plt.contour(X1, X2, Z)
plt.scatter([xhat[0]], [xhat[1]], label="x̂")  # minimum

# plot the gradient
def grad_f1(x):
    return A.T @ A @ x - A.T @ b

def grad_f2(x):
    xt = torch.tensor(x, requires_grad=True)
    f_torch(xt).backward()          # reverse mode: fills in xt.grad
    return xt.grad.numpy()

def grad_f3(x):
    return np.asarray(jax.grad(f_jax)(x))

x = np.array([0.0, 0])
grad_fx = grad_f1(x)

plt.scatter([x[0]], [x[1]], label="x", color="red")  # eval point
plt.arrow(x[0], x[1], 0.1 * grad_fx[0], 0.1 * grad_fx[1], color="red", width=0.05,
          length_includes_head=True, label="0.1⋅∇f(x)")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.gca().set_aspect("equal")
plt.legend()

print(f"Analytical: {grad_f1(x)}")
print(f"PyTorch: {grad_f2(x)}")
print(f"JAX: {grad_f3(x)}")

plt.show()

### Experiments with another function

def f(x):        return np.log(1 + x @ x)
def f_torch(x):  return torch.log(1 + x @ x)
def f_jax(x):    return jnp.log(1 + x @ x)

## Plots the function
Z = np.vectorize(lambda a1, a2: f(np.array([a1, a2])))(X1, X2)

plt.figure(figsize=(5, 5))
plt.contour(X1, X2, Z)

# plot the gradient
def grad_f1(x):
    return 2 / (1 + x @ x) * x

x = np.array([3.0, 0])
grad_fx = grad_f1(x)

plt.scatter([x[0]], [x[1]], label="x", color="red")  # eval point
plt.arrow(x[0], x[1], grad_fx[0], grad_fx[1], color="red", width=0.05,
          length_includes_head=True, label="∇f(x)")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.gca().set_aspect("equal")
plt.legend()

print(f"Analytical: {grad_f1(x)}")
print(f"PyTorch: {grad_f2(x)}")
print(f"JAX: {grad_f3(x)}")

plt.show()
