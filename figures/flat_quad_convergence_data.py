"""
Test convergence using pregenerated quadrature formulae.
"""

from dataclasses import dataclass, asdict
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
from scipy.spatial import Delaunay
from scipy.stats import linregress
import sympy as sym
from tqdm import tqdm

from rbf.geometry import delaunay_covering_radius_stats
from rbf.quadrature import LocalQuad
from rbf.points import UnitSquare
from rbf.rbf import PHS
from utils import hex_stencil_min


DATA_FILE = "data/flat_quad_convergence.json"
SAVE_DATA = True

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


x, y = sym.symbols("x y", real=True)


def cheb7(x):
    return 64 * x**7 - 112 * x**5 + 56 * x**3 - 7 * x


def cheb4(x):
    return 8 * x**4 - 8 * x**2 + 1


def cheb5(x):
    return 16 * x**5 - 20 * x**3 + 5 * x


# sym_func = cheb7(2*x - 1) * cheb7(2*y - 1) + 1,
sym_func = cheb5(2 * x - 1) * cheb4(2 * y - 1) + 1
exact = float(sym.integrate(sym.integrate(sym_func, (x, 0, 1)), (y, 0, 1)))
test_func_str = "$T_5(2x-1) T_4(2y - 1) + 1$"


@dataclass
class Result:
    N: int
    h: float
    rbf: str
    poly_deg: int
    stencil_size: int
    test_func: str
    approx: float
    error: float


rbf = PHS(3)
poly_degs = list(range(5))
stencil_size = hex_stencil_min(21)
print(f"{stencil_size=}")
repeats = 5
Ns = list(map(int, np.logspace(np.log10(5_000), np.log10(100_000), 11)))


test_func = sym.lambdify((x, y), sym_func)
results = []
np.random.seed(0)
for trial in (tqdm_trial := tqdm(range(repeats), position=1, leave=True)):
    for N in (tqdm_N := tqdm(Ns[::-1], position=2, leave=False)):
        tqdm_N.set_description(f"{N=}")
        for poly_deg in (tqdm_poly_deg := tqdm(poly_degs, position=3, leave=False)):
            tqdm_poly_deg.set_description(f"{poly_deg=}")
            points = UnitSquare(
                N=N,
                verbose=True,
                tqdm_kwargs={
                    "position": 4,
                    "leave": False,
                    "desc": "generating points",
                },
            ).points
            mesh = Delaunay(points)
            h, _ = delaunay_covering_radius_stats(mesh)

            quad = LocalQuad(
                points=points,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                verbose=True,
                tqdm_kwargs={
                    "position": 4,
                    "leave": False,
                    "desc": "Calculating weights",
                },
            )

            approx = quad.weights @ test_func(*points.T)
            error = abs(approx - exact) / exact

            result = Result(
                N=N,
                h=h,
                rbf=str(rbf),
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                approx=approx,
                error=error,
                test_func=test_func_str
            )
            results.append(result)
            tqdm_trial.set_description(f"{trial=}, {N=}, {poly_deg=}, {error=:.3E}")

if SAVE_DATA:
    results_dicts = [asdict(result) for result in results]
    with open(DATA_FILE, "w") as f:
        json.dump(results_dicts, f)

if False:
    # for REPL use
    with open(DATA_FILE, "r") as f:
        results = json.load(f)
    results = [Result(**result) for result in results]

plt.figure()
for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [result.h for result in my_res]
    errs = [result.error for result in my_res]
    fit = linregress(np.log(hs), np.log(errs))
    plt.loglog(hs, errs, ".", color=color)
    plt.loglog(
        hs,
        [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
        "-",
        color=color,
        label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    )
plt.legend()
plt.title("Test Function $T_5(2x-1)T_4(2y-1) + 1$")
plt.ylabel("Relative Error")
plt.xlabel("$h$")
# n_max = max(ns)
# n_min = min(ns)
# n_med = int(np.median(ns))
# ns = [n_min, n_med, n_max]
# plt.gca().xaxis.set_major_formatter(ScalarFormatter())
# plt.gca().xaxis.set_minor_formatter(NullFormatter())
# plt.xticks()
# plt.xticks([n**-0.5 for n in ns], [f"${n}^{{-1/2}}$" for n in ns])
