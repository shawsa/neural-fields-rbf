from itertools import product
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sympy as sym


def dist(x, y):
    return sym.sqrt(x**2 + y**2)


# Cartesian sizes
N = 10
CART_FILE = "data/cartesian_stencil_sizes.pickle"
cartesian = list(product(range(-N, N + 1), range(-N, N + 1)))
dists = [dist(x, y) for x, y in cartesian]
dists_unique = sorted(list(set(dists)))
dists_unique = [d for d in dists_unique if d <= N]
counts = [sum(1 for d in dists if d <= my_dist) for my_dist in dists_unique]

plt.figure("Cartesian")
plt.plot(*zip(*cartesian), "k.")
ts = np.linspace(-np.pi, np.pi, 501)
for r in map(float, dists_unique):
    xs = r * np.cos(ts)
    ys = r * np.sin(ts)
    plt.plot(xs, ys, "-")
plt.axis("equal")

with open(CART_FILE, "wb") as f:
    pickle.dump(counts, f)

# hex sizes
N = 10
HEX_FILE = "data/hex_stencil_sizes.pickle"

w = sym.cos(sym.pi / 3) + sym.I * sym.sin(sym.pi / 3)

stencil = [(1 * w**i).expand() for i in range(6)]
points = [0 + 0 * sym.I, 1 + 0*sym.I, w]
center = sum(points) / len(points)
points = [p - center for p in points]
for _ in range(ceil(2 / sqrt(3) * N)):
    new_points = []
    for p in points:
        new_points += [p + q for q in stencil]
    points += new_points
    points = list(set(points))

dists = [sym.Abs(p) for p in points]
dists_unique = sorted(list(set(dists)))
dists_unique = [d for d in dists_unique if d <= N]
counts = [sum(1 for d in dists if d <= my_dist) for my_dist in dists_unique]

xs = [float(sym.re(p)) for p in points]
ys = [float(sym.im(p)) for p in points]


plt.figure("hex")
plt.plot(xs, ys, "k.")
plt.plot(0, 0, "g*")
ts = np.linspace(-np.pi, np.pi, 501)
for r in map(float, dists_unique):
    xs = r * np.cos(ts)
    ys = r * np.sin(ts)
    plt.plot(xs, ys, "-")
plt.axis("equal")

with open(HEX_FILE, "wb") as f:
    pickle.dump(counts, f)
