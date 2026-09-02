import numpy as np
import matplotlib.pyplot as plt


f = lambda x: np.exp(-2*x)
df = lambda x: -2*np.exp(-2*x)
ddf = lambda x: 4*np.exp(-2*x)

x0 = 5

f1 = lambda x: f(x0) + (x - x0) * df(x0)
f2 = lambda x: f(x0) + (x - x0) * df(x0) + 0.5 * (x - x0)**2 * ddf(x0)

d = 4

x = np.linspace(x0 - d/2, x0 + d/2, 200)

plt.plot(x, f(x), label="f", color="black", linewidth=2)
plt.plot(x, f1(x), label="∂f", color="blue", linestyle="--", linewidth=2)
plt.plot(x, f2(x), label="∂²f", color="green", linestyle=":", linewidth=2)
plt.xlabel("x")
plt.legend()
plt.show()
