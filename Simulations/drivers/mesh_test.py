"""
Run convergence Tests for number of points
"""
from drivers.hermite_bump import sym_hermite_bump_poly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from rbf.points.unit_square import UnitSquare
# from rbf.poly_utils import poly_basis_dim
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
import sympy as sym
from tqdm import tqdm


FILE_PREFIX = "drivers/media/mesh_tests/"
# point_set = "random"
# point_set = "cartesian"
point_set = "triangular"

# test_func = "gaussian"
test_func = "hermite"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

match test_func:
    case "gaussian":
        x, y = sym.symbols("x y")
        foo_sym = sym.exp(-200 * (x**2 + y**2))
        foo_single = sym.lambdify((x, y), foo_sym)
        assert foo_single(0.5, 0.5) < 1e-17
        exact = 4 * float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))
    case "hermite":
        x, y, r = sym.symbols("x y r")
        r_sub = sym.sqrt(x**2 + y**2)
        radius = sym.Rational(1, 10)
        smoothness = 4
        foo_r = sym_hermite_bump_poly(r, radius, smoothness)
        # override
        foo_sym = (foo_r * sym.Heaviside(radius - r)).subs(r, r_sub)
        foo_single = sym.lambdify((x, y), foo_sym)
        assert foo_single(0.5, 0.5) < 1e-17
        exact = float(4 * sym.integrate(sym.integrate(r * foo_r, (r, 0, radius)), (y, 0, sym.pi/2)))


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


rbf = PHS(3)
poly_deg = 3
stencil_size = 19
n = 1_000
match point_set:
    case "random":
        print("using random points")
        unit_square = UnitSquare(n, verbose=True, auto_settle=True)
        points = unit_square.points
    case "cartesian":
        print("using Cartesian points")
        n_sqrt = int(np.ceil(np.sqrt(n)))
        X, Y = np.meshgrid(np.linspace(0, 1, n_sqrt),
                           np.linspace(0, 1, n_sqrt))
        points = np.array([X.flatten(), Y.flatten()]).T

    case "triangular":
        print("using triangular points")
        n_x = int(np.ceil(np.sqrt(n / np.sqrt(3))))
        n_y = int(np.ceil(n_x * np.sqrt(3)/2))
        h_x = 1/(n_x - 1)/2
        h_y = 1/(n_y - 1)/2
        X, Y = np.meshgrid(np.linspace(0, 1, n_x),
                           np.linspace(0, 1, n_y))
        X_inner, Y_inner = np.meshgrid(np.linspace(h_x, 1-h_x, n_x-1),
                                       np.linspace(h_y, 1-h_y, n_y-1))
        X = np.append(X, X_inner)
        Y = np.append(Y, Y_inner)
        points = np.array([X.flatten(), Y.flatten()]).T

qf = LocalQuad(points, rbf, poly_deg, stencil_size)

sample_density = 801
X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))
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
ax_error.plot(*points.T, "k.", markersize=.5)
ax_error.triplot(*points.T, qf.mesh.simplices, linewidth=.2)
ax_error.set_xlim(0, 1)
ax_error.set_ylim(0, 1)
error_color = plt.colorbar(error_plot, ax=ax_error)
ax_error.axis("equal")
ax_error.set_title("Quadrature Error for\ntest function centered at $(x_0, y_0)$")
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

plt.savefig(FILE_PREFIX + point_set + "_" + test_func + "_spatial_analysis.png", dpi=300)
