"""
Run convergence Tests for number of points
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gs
import numpy as np
from rbf.quadrature import LocalQuad
from rbf.points import UnitSquare
from rbf.rbf import PHS
from utils import (
    Gaussian,
    hex_grid,
    PeriodicTile,
    quad_test,
    hex_stencil_min,
)


plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

FILE = "media/flat_quad_space"

bump_radius = 0.1
bump = PeriodicTile(Gaussian(bump_radius / 2))

rbf = PHS(3)
poly_deg = 3
stencil_size = hex_stencil_min(21)
print(f"{stencil_size=}")
n = 2_000

sample_density = 401

X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))

print("Generating random points.")
np.random.seed(0)
unit_square = UnitSquare(n, verbose=True, edge_cluster=False)
# cluster top and bottom edges
shift_points = unit_square.inner[:, 1] - 0.5  # just the y
edge_distance = 0.5 - np.max(np.abs(shift_points))
factor = (0.5 - edge_distance / 2) / (0.5 - edge_distance)
unit_square.inner[:, 1] = shift_points * factor + 0.5
rand_points = unit_square.points

print("Computing random point quadrature.")
qf_rand = LocalQuad(rand_points, rbf, poly_deg, stencil_size, verbose=True)
print("Computing random point spatial error.")
err_rand = quad_test(qf=qf_rand, func=bump, X=X, Y=Y, verbose=True)
print(f"max_error={np.max(np.abs(err_rand)):.3E}")

hex_points = hex_grid(n)
print("Computing hex point quadrature.")
qf_hex = LocalQuad(hex_points, rbf, poly_deg, stencil_size, verbose=True)
print("Computing hex point spatial error.")
err_hex = quad_test(qf=qf_hex, func=bump, X=X, Y=Y, verbose=True)
print(f"max_error={np.max(np.abs(err_hex)):.3E}")

#############
#
# Figure
#
#############

print("Generating Figure.")
# figsize = (8, 10)
figsize = (8, 6)
fig = plt.figure("quad_space_error", figsize=figsize)

# grid = gs.GridSpec(23, 24)
grid = gs.GridSpec(15, 24)

lcol = slice(0, 8)
lbar = slice(8, 9)
rcol = slice(12, 20)
rbar = slice(20, 21)

weights_col = slice(9, 17)
hist_col = slice(17, 24)

row1 = slice(0, 7)
row2 = slice(8, 15)
row3 = slice(16, 23)

y_label_loc = (-0.07, 0.5)
x_label_loc = (0.5, -0.03)

# test func
ax_test_func = fig.add_subplot(grid[row1, lcol])
center = np.array([0.35, 0.75])
exact = bump.integrate()
approx = sum(
    w * bump(*(point - center)) for w, point in zip(qf_rand.weights, rand_points)
)
error = (approx - exact) / exact
ax_test_func.triplot(*rand_points.T, qf_rand.mesh.simplices, linewidth=0.2)
test_func = ax_test_func.pcolormesh(
    X, Y, bump(X - center[0], Y - center[1]), cmap="viridis"
)
ax_test_func.set_xlim(0, 1)
ax_test_func.set_ylim(0, 1)
ax_test_func.axis("equal")
ax_test_func.set_title("$f_{%s}$" % str(tuple(center)))
ax_test_func.axis("off")

ax_test_func.text(*x_label_loc, "$x$", transform=ax_test_func.transAxes)
ax_test_func.text(*y_label_loc, "$y$", transform=ax_test_func.transAxes)

# weights
ax_weights = fig.add_subplot(grid[row1, weights_col])
neg_mask = qf_rand.weights < 0
neg_style = {
    "marker": ".",
    "linestyle": "",
    "markerfacecolor": (0.0, 0.0, 0.0, 1.0),
    "markeredgecolor": "k",
}
weight_color = ax_weights.scatter(
    *rand_points.T, c=qf_rand.weights, cmap="viridis", s=1
)
ax_weights.plot(*rand_points[neg_mask].T, **neg_style)
ax_weights.set_ylim(0, 1)
ax_weights.axis("equal")
ax_weights.axis("off")
ax_weights.set_title("Weights")

ax_weights.text(*y_label_loc, "$y$", transform=ax_weights.transAxes)
ax_weights.text(*x_label_loc, "$x$", transform=ax_weights.transAxes)

# ax_weights_bar = fig.add_subplot(grid[row1, 8])
# weight_color_bar = plt.colorbar(weight_color, cax=ax_weights_bar)

# histogram
ax_hist = fig.add_subplot(grid[row1, hist_col])
_, bins, patches = ax_hist.hist(qf_rand.weights, bins=25, orientation="horizontal")
cnorm = Normalize(np.min(qf_rand.weights), np.max(qf_rand.weights))
for val, patch in zip(bins, patches):
    patch.set_facecolor(plt.cm.viridis(cnorm(val)))

ax_hist.axis("off")
ax_hist.text(20, 0, "0")
ax_hist.text(20, 5e-4, "0.0005")
ax_hist.text(20, 9e-4, "0.0009")


# rand error
ax_rand_space = fig.add_subplot(grid[row2, lcol])
error_plot = ax_rand_space.pcolormesh(X, Y, err_rand, cmap="viridis")
ax_rand_space.triplot(*rand_points.T, qf_rand.mesh.simplices, linewidth=0.2)
# ax_rand_space.plot(*center, "k*")
ax_rand_space.plot(*rand_points.T, "k.", markersize=0.5)
ax_rand_space.set_xlim(0, 1)
ax_rand_space.set_ylim(0, 1)

ax_rand_error_cbar = fig.add_subplot(grid[row2, lbar])
error_color = plt.colorbar(error_plot, cax=ax_rand_error_cbar)

ax_rand_space.axis("equal")
ax_rand_space.set_title("Relative Error")

ax_rand_space.axis("off")
ax_rand_space.text(*x_label_loc, "$x_0$", transform=ax_rand_space.transAxes)
ax_rand_space.text(*y_label_loc, "$y_0$", transform=ax_rand_space.transAxes)

# rand log error
ax_rand_log_space = fig.add_subplot(grid[row2, rcol])
log_error_plot = ax_rand_log_space.pcolormesh(
    X, Y, np.log10(np.abs(err_rand)), cmap="viridis"
)
ax_rand_log_space.set_xlim(0, 1)
ax_rand_log_space.set_ylim(0, 1)

ax_rand_log_space_cbar = fig.add_subplot(grid[row2, rbar])
log_error_color = plt.colorbar(log_error_plot, cax=ax_rand_log_space_cbar)
ax_rand_log_space.axis("equal")
ax_rand_log_space.set_title("Log10 Relative Error")
ax_rand_log_space.set_xlabel("$x_0$")

ax_rand_log_space.axis("off")
ax_rand_log_space.text(*x_label_loc, "$x_0$", transform=ax_rand_log_space.transAxes)
ax_rand_log_space.text(*y_label_loc, "$y_0$", transform=ax_rand_log_space.transAxes)

# hex error
# ax_hex_space = fig.add_subplot(grid[row3, lcol])
# error_plot = ax_hex_space.pcolormesh(X, Y, err_hex, cmap="viridis")
# ax_hex_space.triplot(*hex_points.T, qf_hex.mesh.simplices, linewidth=0.2)
# ax_hex_space.plot(*hex_points.T, "k.", markersize=0.5)
# ax_hex_space.set_xlim(0, 1)
# ax_hex_space.set_ylim(0, 1)

# ax_hex_space_bar = fig.add_subplot(grid[row3, lbar])
# error_color = plt.colorbar(error_plot, cax=ax_hex_space_bar)
# ax_hex_space.axis("equal")
# ax_hex_space.axis("off")
# ax_hex_space.set_title("Relative Error")

# ax_hex_space.text(*x_label_loc, "$x_0$", transform=ax_hex_space.transAxes)
# ax_hex_space.text(*y_label_loc, "$y_0$", transform=ax_hex_space.transAxes)

# log hex error
# ax_hex_log_space = fig.add_subplot(grid[row3, rcol])
# log_error_plot = ax_hex_log_space.pcolormesh(
#     X, Y, np.log10(np.abs(err_hex)), cmap="viridis"
# )
# ax_hex_log_space.set_xlim(0, 1)
# ax_hex_log_space.set_ylim(0, 1)

# ax_hex_log_space_bar = fig.add_subplot(grid[row3, rbar])
# log_error_color = plt.colorbar(log_error_plot, cax=ax_hex_log_space_bar)
# ax_hex_log_space.axis("equal")
# ax_hex_log_space.axis("off")
# ax_hex_log_space.set_title("Log10 Relative Error")

# ax_hex_log_space.text(*y_label_loc, "$y_0$", transform=ax_hex_log_space.transAxes)
# ax_hex_log_space.text(*x_label_loc, "$x_0$", transform=ax_hex_log_space.transAxes)


# Panel labels
subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "STIXGeneral",
    "usetex": True,
}
for ax, label in zip(
    [
        ax_test_func,
        ax_weights,
        ax_rand_space,
        ax_rand_log_space,
        # ax_hex_space,
        # ax_hex_log_space,
    ],
    "ABCDEFGH",
):
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        transform=ax.transAxes,
        **subplot_label_font,
    )

# plt.suptitle("RBF-QF on the Unit Square")

grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig(FILE + ".png", dpi=300, bbox_inches="tight")
