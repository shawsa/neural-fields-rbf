from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from rbf.geometry import delaunay_covering_radius_stats
from rbf.points.unit_square import UnitSquare

from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from scipy.spatial import Delaunay
from scipy.stats import linregress
from tqdm import tqdm

from utils import (
    Gaussian,
    PeriodicTile,
    quad_test,
    hex_stencil_min,
)

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

bump_radius = 0.1
bump = PeriodicTile(Gaussian(bump_radius / 2))
exact = bump.integrate()


repeats = 20
rbf = PHS(3)
poly_degs = [3]  # list(range(3))
stencil_size = hex_stencil_min(50)

x0_bounds = (0.3, 0.7)


n_min = 500
n_max = 1_000
num_ns = 2
ns = list(reversed(np.logspace(np.log10(n_min), np.log10(n_max), num_ns, dtype=int)))

results = []
Result = namedtuple("Result", ["n", "h_max", "h_ave", "poly_deg", "error"])
for n in (tqdm_obj := tqdm(ns, leave=False, position=0)):
    for _ in tqdm(range(repeats), leave=False, position=1):
        unit_square = UnitSquare(n, verbose=False, auto_settle=True)
        points = unit_square.points
        mesh = Delaunay(points)
        h_max, h_ave = delaunay_covering_radius_stats(mesh)
        X, Y = np.meshgrid(*(np.linspace(*x0_bounds, int(10 / h_max)),) * 2)
        num_tests = len(X.ravel())
        for poly_deg in tqdm(poly_degs, leave=False, position=2):
            qf = LocalQuad(points, rbf, poly_deg, stencil_size)
            error = 0
            for x0, y0 in tqdm(
                zip(X.ravel(), Y.ravel()),
                total=num_tests,
                leave=False,
                position=3,
            ):
                approx = qf.weights @ bump(points[:, 0] - x0, points[:, 1] - y0)
                sample_error = np.abs(approx - exact) / exact
                error = max(error, sample_error)

            tqdm_obj.set_description(f"{n=} | {poly_deg=} | {error=:.3E}")
            results.append(
                Result(n=n, h_max=h_max, h_ave=h_ave, poly_deg=poly_deg, error=error)
            )


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
colors = ["r", "b", "g", "c", "m", "y"] * 2
for poly_deg, color in zip(poly_degs, colors):
    hs = [result.h_max for result in results if result.poly_deg == poly_deg]
    errors = [result.error for result in results if result.poly_deg == poly_deg]
    fit = linregress(np.log(hs), np.log(errors))
    ax.loglog(
        hs,
        errors,
        ".",
        color=color,
    )
    plt.loglog(
        hs,
        [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
        "-",
        color=color,
        label=f"deg={poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$h_{max}$")
# ax.set_xlabel("$h_{max}$")
ax.set_ylabel("Relative Error")
ax.legend()
ax.grid()

# plt.figure()
# plt.triplot(*points.T, mesh.simplices)
# circum_radii = []
# centroids = []
# for tri_indices in mesh.simplices:
#     tri_points = mesh.points[tri_indices]
#     centroids.append(triangle(tri_points).centroid)
#     circum_radii.append(circumradius(tri_points))
# centroids = np.array(centroids)
# plt.scatter(*centroids.T, c=circum_radii)
# plt.colorbar(label="$h$")
# plt.savefig("media/h_spacing.png")
