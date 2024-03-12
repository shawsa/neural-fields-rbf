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
    hex_stencil_min,
)


plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


FILE_PREFIX = "media/poster_space_"
SAVE_FIGURES = False

rbf = PHS(3)
poly_deg = 3
np.random.seed(0)
stencil_size = hex_stencil_min(21)
n = 2_000

sample_density = 801
bump_radius = 0.1

X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))

grid = gs.GridSpec(2, 2)
fig = plt.figure(figsize=(7, 6))

for row, (point_set, point_generator) in enumerate([
    ("Scattered Nodes", lambda n: random_points(n, verbose=True)),
    ("Regular Grid", hex_grid),
]):
    print("Generating points and QF")
    print(f"{point_set=}")
    points = point_generator(n)
    qf = LocalQuad(points, rbf, poly_deg, stencil_size)
    for test_func, bump in [
        ("gauss", PeriodicTile(Gaussian(bump_radius / 2))),
    ]:
        print(f"{test_func=}")
        Z = quad_test(qf=qf, func=bump, X=X, Y=Y, verbose=True)
        ax_error = fig.add_subplot(grid[row, 0])
        error_plot = ax_error.pcolormesh(X, Y, Z, cmap="jet")
        ax_error.plot(*points.T, "k.", markersize=0.5)
        ax_error.triplot(*points.T, qf.mesh.simplices, linewidth=0.2)
        error_color = plt.colorbar(error_plot, ax=ax_error)
        ax_error.axis("equal")
        ax_error.set_title(point_set + " Error")
        ax_error.set_xlabel("$x_0$")
        ax_error.set_ylabel("$y_0$")
        ax_error.set_xticks([])
        ax_error.set_yticks([])
        # ax_error.axis("off")

        ax_log_error = fig.add_subplot(grid[row, 1])
        log_error_plot = ax_log_error.pcolormesh(X, Y, np.log10(np.abs(Z)), cmap="jet")
        log_error_color = plt.colorbar(log_error_plot, ax=ax_log_error)
        ax_log_error.axis("equal")
        # ax_log_error.axis("off")
        ax_log_error.set_xlabel("$x_0$")
        ax_log_error.set_xticks([])
        ax_log_error.set_yticks([])
        ax_log_error.set_title("Log10 Relative Error")
        grid.tight_layout(fig)

        plt.pause(1e-3)

if SAVE_FIGURES:
    plt.savefig(FILE_PREFIX + "combined.png", dpi=300)
