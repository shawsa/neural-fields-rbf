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


FILE = "drivers/media/convergence"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


repeats = 20
hs = np.logspace(np.log2(1e-1), np.log2(1e-2), 10, base=2)
rbf = PHS(3)
poly_degs = list(range(7))
stencil_size = 30


ns = list(map(hex_limit_density, hs))

x, y = sym.symbols("x y")
foo_sym = sym.cos(5*x) * sym.sin(7*y) + 1
foo = sym.lambdify((x, y), foo_sym)
exact = float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))

results = []
Result = namedtuple("Result", ["n", "h_max", "h_ave", "poly_deg", "error"])
tqdm_obj = tqdm(ns, leave=False, position=0)
for n in tqdm_obj:
    for _ in tqdm(range(repeats), leave=False, position=1):
        unit_square = UnitSquare(n, verbose=False, auto_settle=True)
        points = unit_square.points
        mesh = Delaunay(points)
        h_max, h_ave = delaunay_covering_radius_stats(mesh)
        fs = foo(*points.T)
        for poly_deg in tqdm(poly_degs, leave=False, position=2):
            qf = LocalQuad(points, rbf, poly_deg, stencil_size)
            approx = qf.weights @ fs
            error = abs((approx - exact) / exact)
            tqdm_obj.set_description(f"{n=} | {poly_deg=} | {error=:.3E}")
            results.append(Result(
                n=n,
                h_max=h_max,
                h_ave=h_ave,
                poly_deg=poly_deg,
                error=error
            ))


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
colors = ["r", "b", "g", "c", "m", "y"] * 2
for poly_deg, color in zip(poly_degs, colors):
    hs = [result.h_ave for result in results if result.poly_deg == poly_deg]
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
ax.set_xlabel("$h_{ave}$")
# ax.set_xlabel("$h_{max}$")
ax.set_ylabel("Relative Error")
ax.set_title(f"Integral of ${sym.latex(foo_sym)}$ over $[0, 1]^2$")
ax.legend()
ax.grid()

for ext in [".pdf", ".png"]:
    plt.savefig(FILE + ext)

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
