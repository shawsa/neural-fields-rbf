"""
Run convergence Tests for number of points
"""
from collections import namedtuple
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import numpy.linalg as la
from rbf.geometry import circumradius, delaunay_covering_radius_stats, triangle
from rbf.points.unit_square import UnitSquare, hex_limit_density

# from rbf.poly_utils import poly_basis_dim
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from scipy.spatial import Delaunay
from scipy.stats import linregress
import sympy as sym
from tqdm import tqdm


FILE_PREFIX = "drivers/media/mesh_tests/"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


x, y = sym.symbols("x y")
foo_sym = sym.exp(-200 * (x**2 + y**2))
foo_single = sym.lambdify((x, y), foo_sym)
assert foo_single(0.5, 0.5) < 1e-17
exact = 4 * float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))


def foo_tile(x, y):
    ret = np.zeros_like(x)
    for x_0 in [-1, 0, 1, 2]:
        for y_0 in [-1, 0, 1, 2]:
            ret += foo_single(x - x_0, y - y_0)
    return ret


def foo(
    x: np.ndarray[float], y: np.ndarray[float], x0: float = 0, y0: float = 0
) -> np.ndarray[float]:
    return foo_tile(x - x0, y - y0)


plt.figure("f(x, y)")
X, Y = np.meshgrid(np.linspace(0, 1, 2_000), np.linspace(0, 1, 2_000))
plt.pcolormesh(X, Y, foo(X, Y))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis("equal")

rbf = PHS(3)
poly_deg = 3
stencil_size = 21
n = 1_000
unit_square = UnitSquare(n, verbose=True, auto_settle=True)
points = unit_square.points
qf = LocalQuad(points, rbf, poly_deg, stencil_size)

sample_density = 401
X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))
Z = np.zeros_like(X).flatten()
for index, (x0, y0) in tqdm(
    enumerate(zip(X.ravel(), Y.ravel())), total=sample_density**2
):
    fs = foo(*points.T, x0=x0, y0=y0)
    approx = qf.weights @ fs
    error = abs((approx - exact) / exact)
    Z[index] = error
Z = Z.reshape(X.shape)
plt.figure("Quad Error")
plt.pcolormesh(X, Y, np.log10(np.abs(Z)), cmap="jet")
# plt.plot(*points.T, "r.")
# plt.triplot(*points.T, qf.mesh.simplices)
plt.colorbar()
plt.axis("equal")


plt.figure("Weights")
plt.scatter(*points.T, c=qf.weights, cmap="jet")
plt.colorbar()
plt.axis("equal")

plt.figure("Weights Histogram")
plt.hist(qf.weights, bins=25)
