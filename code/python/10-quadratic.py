import numpy as np
import matplotlib.pyplot as plt

### Choose vectors and variables

e1 = np.array([1, 1])
e1 = e1 / np.linalg.norm(e1)
l1 = 0.2
e2 = np.array([1, -1])
e2 = e2 / np.linalg.norm(e2)
l2 = 1
assert abs(e1 @ e2) <= 1e-5    # check that they are orthogonal

print("Start with a sum of squares: ")
print("%.1f (%.1f x₁ %+.1f x₂)²  + %.1f (%.1f x₁ %+.1f x₂)²" % (l1, *e1, l2, *e2))


### Construct quadratic function

Q = np.column_stack((e1, e2))
L = np.diag([l1, l2])
A = Q @ L @ Q.T

print("Turn to a quadratic function: ")
print("%.1f x₁² %+.1f x₁x₂ %+.1f x₂²" % (A[0, 0], A[0, 1] + A[1, 0], A[1, 1]))

### Recover the sum of quadratics

l, Q = np.linalg.eigh(A)   # eigenvectors are unique only up to sign

print("Recover the sum of squares: ")
print("%.1f (%.1f x₁ %+.1f x₂)²  + %.1f (%.1f x₁ %+.1f x₂)²" %
      (l[0], *Q[:, 0], l[1], *Q[:, 1]))

### Plot the quadratic function

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
# f(x) = xᵀ A x evaluated on the grid
Z = A[0, 0] * X1**2 + (A[0, 1] + A[1, 0]) * X1 * X2 + A[1, 1] * X2**2

plt.figure(figsize=(5, 5))
plt.contour(X1, X2, Z)
plt.arrow(0, 0, l1 * e1[0], l1 * e1[1], color="red", width=0.02,
          length_includes_head=True, label="λ₁ ⋅ e₁")
plt.arrow(0, 0, l2 * e2[0], l2 * e2[1], color="blue", width=0.02,
          length_includes_head=True, label="λ₂ ⋅ e₂")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.gca().set_aspect("equal")
plt.legend()
plt.show()

### And a surface plot

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
ax.plot_surface(X1, X2, Z, cmap="viridis")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
plt.show()
