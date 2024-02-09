"""
Run convergence Tests for number of points
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from math import ceil
import numpy as np
from rbf.points import UnitSquare
from rbf.poly_utils import poly_basis_dim
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
import sympy as sym
from tqdm import tqdm


FILE = "drivers/media/convergence"

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


ns = [2**index for index in range(4, 8)]
rbf = PHS(3)
poly_degs = list(range(-1, 4))
min_stencil_size = 7
stencil_size_factor = 1.5

x, y = sym.symbols("x y")
foo_sym = sym.cos(x)**2 - y*sym.exp(y)
foo = sym.lambdify((x, y), foo_sym)
exact = float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))

results = {}

for n in tqdm(ns):
    print("Generating Points")
    points = UnitSquare(n, verbose=True).points
    fs = foo(*points.T)
    for poly_deg in tqdm(poly_degs):
        stencil_size = max(
                ceil(stencil_size_factor * poly_basis_dim(2, poly_deg)),
                min_stencil_size)
        qf = LocalQuad(points, rbf, poly_deg, stencil_size)
        approx = qf.weights @ fs
        error = abs((approx - exact)/exact)
        results[n, poly_deg] = (points, qf, error)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
for poly_deg in poly_degs:
    errors = [results[n, poly_deg][2] for n in ns]
    order = -np.log(errors[0]/errors[-1]) / np.log(ns[0] / ns[-1])
    ax.loglog(ns, errors, ".-", label=f"deg={poly_deg}~$\\mathcal{{O}}({order:.2f})$")

ax.minorticks_off()
ax.set_xticks(ns, ns)
ax.set_yscale("log", base=2)
ax.set_xlabel("$\\sqrt{N}$")
ax.set_ylabel("Relative Error")
ax.set_title(f"Integral of ${sym.latex(foo_sym)}$ over $[0, 1]^2$")
ax.legend()
ax.grid()

for ext in [".pdf", ".png"]:
    plt.savefig(FILE + ext)
