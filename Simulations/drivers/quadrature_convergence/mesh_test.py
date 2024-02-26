"""
Run convergence Tests for number of points
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from utils import (
    Gaussian,
    HermiteBump,
    random_points,
    hex_grid,
    PeriodicTile,
    quad_test,
)
from tqdm import tqdm


plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


FILE_PREFIX = "media/mesh_test_"
SAVE_FIGURES = False

rbf = PHS(3)
poly_deg = 3
stencil_size = 21
n = 2_000

sample_density = 401
bump_radius = 0.1

X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))
for point_set, point_generator in [
    ("random", lambda n: random_points(n, verbose=True)),
    ("hex", hex_grid),
]:
    print("Generating points and QF")
    print(f"{point_set=}")
    points = point_generator(n)
    qf = LocalQuad(points, rbf, poly_deg, stencil_size)
    for test_func, bump in [
        ("gauss", PeriodicTile(Gaussian(bump_radius / 2))),
        ("hermite", PeriodicTile(HermiteBump(order=3, radius=bump_radius))),
    ]:
        print(f"{test_func=}")
        Z = quad_test(qf=qf, func=bump, X=X, Y=Y, verbose=True)
        grid = gs.GridSpec(2, 2)
        fig = plt.figure("_".join([point_set, test_func]), figsize=(8, 8))

        plt.suptitle(f"{point_set=}, {test_func=}")
        ax_error = fig.add_subplot(grid[0])
        error_plot = ax_error.pcolormesh(X, Y, Z, cmap="jet")
        ax_error.plot(*points.T, "k.", markersize=0.5)
        ax_error.triplot(*points.T, qf.mesh.simplices, linewidth=0.2)
        ax_error.set_xlim(0, 1)
        ax_error.set_ylim(0, 1)
        error_color = plt.colorbar(error_plot, ax=ax_error)
        ax_error.axis("equal")
        ax_error.set_title(
            "Quadrature Error for\ntest function centered at $(x_0, y_0)$"
        )
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
        weight_color = ax_weights.scatter(
            *points.T, c=qf.weights, cmap="jet", s=1, zorder=20
        )
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
        plt.show()

        if SAVE_FIGURES:
            plt.savefig(FILE_PREFIX + point_set + "_" + test_func, dpi=300)
