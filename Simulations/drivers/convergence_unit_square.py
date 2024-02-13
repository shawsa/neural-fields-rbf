"""
Run convergence Tests for number of points
"""
import matplotlib.pyplot as plt

from math import ceil
import numpy as np
import numpy.linalg as la
from rbf.geometry import circumradius, delaunay_covering_radius, triangle
from rbf.points import UnitSquare

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


repeats = 5
# ns = [ceil(2**(index/2) for index in range(3, 8)]
num_ns = 10
ns = [int(n) for n in np.logspace(np.log2(7), np.log2(80), num_ns, base=2)]
rbf = PHS(3)
poly_degs = list(range(7))
stencil_size = 35

x, y = sym.symbols("x y")
foo_sym = sym.exp(x*y)
foo = sym.lambdify((x, y), foo_sym)
exact = float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))

results = []
tqdm_obj = tqdm(ns, leave=False, position=0)
for n in tqdm_obj:
    unit_square = UnitSquare(n, verbose=False, auto_settle=False)
    for _ in tqdm(range(repeats), leave=False, position=1):
        unit_square.auto_settle()
        points = unit_square.points
        mesh = Delaunay(points)
        h = delaunay_covering_radius(mesh)
        fs = foo(*points.T)
        for poly_deg in tqdm(poly_degs, leave=False, position=2):
            qf = LocalQuad(points, rbf, poly_deg, stencil_size)
            approx = qf.weights @ fs
            error = abs((approx - exact) / exact)
            tqdm_obj.set_description(f"{n=} | {poly_deg=} | {error=:.3E}")
            results.append((n, poly_deg, h, error))


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
colors = ["r", "b", "g", "c", "m", "y"] * 2
for poly_deg, color in zip(poly_degs, colors):
    hs = []
    errors = []
    for n, poly_deg2, h, error in results:
        if poly_deg2 != poly_deg:
            continue
        hs.append(h)
        errors.append(error)
    # order = np.log(errors[0] / errors[-1]) / np.log(hs[0] / hs[-1])
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

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("$h$")
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
