"""
Run convergence Tests for number of points
"""
import matplotlib.pyplot as plt
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
rand_points = UnitSquare(n, verbose=True).points
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
figsize = (8.5, 11)
fig = plt.figure("quad_space_error", figsize=figsize)
grid = gs.GridSpec(3, 2)

# Panel A
ax_rand_space = fig.add_subplot(grid[0, 0])
error_plot = ax_rand_space.pcolormesh(X, Y, err_rand, cmap="jet")
ax_rand_space.triplot(*rand_points.T, qf_rand.mesh.simplices, linewidth=0.2)
ax_rand_space.plot(*rand_points.T, "k.", markersize=0.5)
ax_rand_space.set_xlim(0, 1)
ax_rand_space.set_ylim(0, 1)
error_color = plt.colorbar(error_plot, ax=ax_rand_space)
ax_rand_space.axis("equal")
ax_rand_space.set_title("Relative Error")
ax_rand_space.set_xlabel("$x_0$")
ax_rand_space.set_ylabel("$y_0$")

# Panel B
ax_rand_log_space = fig.add_subplot(grid[0, 1])
log_error_plot = ax_rand_log_space.pcolormesh(
    X, Y, np.log10(np.abs(err_rand)), cmap="jet"
)
ax_rand_log_space.set_xlim(0, 1)
ax_rand_log_space.set_ylim(0, 1)
log_error_color = plt.colorbar(log_error_plot, ax=ax_rand_log_space)
ax_rand_log_space.axis("equal")
ax_rand_log_space.set_title("Log10 Relative Error")
ax_rand_log_space.set_xlabel("$x_0$")

# Panel C
ax_weights = fig.add_subplot(grid[1, 0])
neg_mask = qf_rand.weights < 0
neg_style = {
    "marker": ".",
    "linestyle": "",
    "markerfacecolor": "w",
    "markeredgecolor": "k",
}
ax_weights.plot(*rand_points[neg_mask].T, **neg_style, zorder=-20)
weight_color = ax_weights.scatter(
    *rand_points.T, c=qf_rand.weights, cmap="jet", s=1, zorder=20
)
weight_color_bar = plt.colorbar(weight_color, ax=ax_weights)
ax_weights.axis("equal")
ax_weights.set_title("Weights")
ax_weights.set_xlabel("$x$")
ax_weights.set_ylabel("$y$")

# Panel D
ax_hist = fig.add_subplot(grid[1, 1])
ax_hist.hist(qf_rand.weights, bins=25)
ax_hist.set_title("Histogram of Weights")
ax_hist.set_xlabel("weights")

# Panel E
ax_hex_space = fig.add_subplot(grid[2, 0])
error_plot = ax_hex_space.pcolormesh(X, Y, err_hex, cmap="jet")
ax_hex_space.triplot(*hex_points.T, qf_hex.mesh.simplices, linewidth=0.2)
ax_hex_space.plot(*hex_points.T, "k.", markersize=0.5)
ax_hex_space.set_xlim(0, 1)
ax_hex_space.set_ylim(0, 1)
error_color = plt.colorbar(error_plot, ax=ax_hex_space)
ax_hex_space.axis("equal")
ax_hex_space.set_title("Relative Error")
ax_hex_space.set_xlabel("$x_0$")
ax_hex_space.set_ylabel("$y_0$")

# Panel F
ax_hex_log_space = fig.add_subplot(grid[2, 1])
log_error_plot = ax_hex_log_space.pcolormesh(
    X, Y, np.log10(np.abs(err_hex)), cmap="jet"
)
ax_hex_log_space.set_xlim(0, 1)
ax_hex_log_space.set_ylim(0, 1)
log_error_color = plt.colorbar(log_error_plot, ax=ax_hex_log_space)
ax_hex_log_space.axis("equal")
ax_hex_log_space.set_title("Log10 Relative Error")
ax_hex_log_space.set_xlabel("$x_0$")

# Panel labels
subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "sans",
    "usetex": False,
}
for ax, label in [
    (ax_rand_space, "A"),
    (ax_rand_log_space, "B"),
    (ax_weights, "C"),
    (ax_hist, "D"),
    (ax_hex_space, "E"),
    (ax_hex_log_space, "F"),
]:
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        transform=ax.transAxes,
        **subplot_label_font,
    )


grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig(FILE + ".png", dpi=300)
