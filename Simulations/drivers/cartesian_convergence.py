"""
Run convergence Tests for number of points
"""
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
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


FILE_PREFIX = "drivers/media/"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


def cartesian_grid(n: int):
    return np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))


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


sample_density = 21
X_sample, Y_sample = cartesian_grid(sample_density)
X_sample = X_sample.flatten()
Y_sample = Y_sample.flatten()


def error_max_ave(points, weights):
    e_max = 0
    e_sum = 0
    for x0, y0 in zip(X_sample, Y_sample):
        fs = foo(*points.T, x0=x0, y0=y0)
        approx = qf.weights @ fs
        error = (approx - exact) / exact
        print(error)
        e_max = max([e_max, error])
        e_sum += error
    e_ave = e_sum / len(X_sample)
    return e_max, e_ave


rbf = PHS(3)
poly_deg = 3
stencil_size = 21
X, Y = cartesian_grid(50)
points = np.array([X.flatten(), Y.flatten()]).T
qf = LocalQuad(points, rbf, poly_deg, stencil_size)

Z = np.zeros_like(X).flatten()
for index, (x0, y0) in tqdm(
    enumerate(zip(X.ravel(), Y.ravel())), total=sample_density**2
):
    fs = foo(*points.T, x0=x0, y0=y0)
    approx = qf.weights @ fs
    error = (approx - exact) / exact
    Z[index] = error
Z = Z.reshape(X.shape)

grid = gs.GridSpec(2, 2)
fig = plt.figure(figsize=(8, 8))

ax_error = fig.add_subplot(grid[0])
error_plot = ax_error.pcolormesh(X, Y, Z, cmap="jet")
ax_error.plot(*points.T, "k.", markersize=0.5)
ax_error.triplot(*points.T, qf.mesh.simplices, linewidth=0.2)
ax_error.set_xlim(0, 1)
ax_error.set_ylim(0, 1)
error_color = plt.colorbar(error_plot, ax=ax_error)
ax_error.axis("equal")
ax_error.set_title("Quadrature Error for\nGaussian centered at $(x_0, y_0)$.")
ax_error.set_xlabel("$x_0$")
ax_error.set_ylabel("$y_0$")
error_color.set_label("Relative Error")

ax_log_error = fig.add_subplot(grid[1])
log_error_plot = ax_log_error.pcolormesh(X, Y, np.log10(np.abs(Z)), cmap="jet")
ax_log_error.set_xlim(0, 1)
ax_log_error.set_ylim(0, 1)
log_error_color = plt.colorbar(log_error_plot, ax=ax_log_error)
ax_log_error.axis("equal")
ax_log_error.set_title("Log Error")
ax_log_error.set_xlabel("$x_0$")
ax_log_error.set_ylabel("$y_0$")
log_error_color.set_label("Log10 Relative Error")

ax_weights = fig.add_subplot(grid[2])
neg_mask = qf.weights < 0
neg_style = {
    "marker": ".",
    "linestyle": "",
    "markerfacecolor": "w",
    "markeredgecolor": "k",
}
ax_weights.plot(*points[neg_mask].T, **neg_style, zorder=-20)
weight_color = ax_weights.scatter(*points.T, c=qf.weights, cmap="jet", s=1, zorder=20)
weight_color_bar = plt.colorbar(weight_color, ax=ax_weights)
ax_weights.axis("equal")
ax_weights.set_title("Weights")
ax_weights.set_xlabel("$x$")
ax_weights.set_ylabel("$y$")
weight_color_bar.set_label("Weight")

ax_hist = fig.add_subplot(grid[3])
ax_hist.hist(qf.weights, bins=25)
ax_hist.set_title("Histogram of Weights")
ax_hist.set_xlabel("weights")


grid.tight_layout(fig)

plt.savefig(FILE_PREFIX + "spatial_analysis.png", dpi=300)
