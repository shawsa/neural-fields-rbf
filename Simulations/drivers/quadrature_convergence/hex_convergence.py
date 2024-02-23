"""
Run convergence Tests for number of points
"""
from collections import namedtuple
from math import ceil, sqrt
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
import pickle
from rbf.points.unit_square import hex_limit_density
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from scipy.stats import linregress
from tqdm import tqdm
from utils import HermiteBump, hex_grid, hex_stencil_min


DATA_FILE = "data/hex_convergence_hermite.pickle"

rbf = PHS(3)
poly_degs = range(5)
stencil_size_factor = 1.2
h_targets = np.logspace(-1, -2, 21)

sample_density = 801
bump_order = 3
bump_radius = 0.1

inner_dist = 0.2


def get_stencil_size(poly_deg: int) -> int:
    return hex_stencil_min(ceil(stencil_size_factor * (poly_deg + 1) * (poly_deg + 2)))


bump = HermiteBump(order=bump_order, radius=bump_radius)

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

foo_single = bump
assert foo_single(0.5, 0.5) < 1e-17
exact = bump.integrate()


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


X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))
inner = np.logical_and(X > inner_dist, X < 1 - inner_dist)
inner = np.logical_and(inner, Y > inner_dist)
inner = np.logical_and(inner, Y < 1 - inner_dist)
X = X.flatten()
Y = Y.flatten()
inner = inner.flatten()

ErrorStats = namedtuple("ErrorStats", ["min", "max", "median", "average"])
Result = namedtuple(
    "Result", ["n", "poly_deg", "stencil_size", "error_stats", "inner_error_stats"]
)


def calculate_error_stats(errors: np.ndarray[float]) -> ErrorStats:
    errors = np.abs(errors)
    return ErrorStats(
        min=np.min(errors),
        max=np.max(errors),
        median=np.median(errors),
        average=np.average(errors),
    )


results = []
for h_target in (hs_prog := tqdm(h_targets, position=0)):
    points = hex_grid(hex_limit_density(h_target))
    n = len(points)
    hs_prog.set_description(f"{n=}")
    for poly_deg in (deg_prog := tqdm(poly_degs, leave=False, position=1)):
        deg_prog.set_description(f"{poly_deg=}")
        stencil_size = get_stencil_size(poly_deg)
        qf = LocalQuad(points, rbf, poly_deg, stencil_size)
        errors = np.zeros_like(X)
        for index, (x0, y0) in tqdm(
            enumerate(zip(X, Y)),
            total=len(X),
            position=2,
            leave=False,
        ):
            fs = foo(*points.T, x0=x0, y0=y0)
            approx = qf.weights @ fs
            error = (approx - exact) / exact
            errors[index] = error
        result = Result(
            n=n,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
            error_stats=calculate_error_stats(errors),
            inner_error_stats=calculate_error_stats(errors[inner]),
        )
        results.append(result)
        hs_prog.set_description(f"{n=}, error={result.error_stats.max:.3E}")


HexConvergenceData = namedtuple("HexConvergenceData", [
        "bump_order",
        "bump_radius",
        "inner_distance",
        "sample_density",
        "rbf",
        "results",
])

data = HexConvergenceData(
        bump_order=bump_order,
        bump_radius=bump_radius,
        inner_distance=inner_dist,
        sample_density=sample_density,
        rbf=rbf,
        results=results,
)

with open(DATA_FILE, "wb") as f:
    pickle.dump(data, f)

if False:
    # for REPL use
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
        results = data.results


# make plot

for err_str, err_func in [
    ("max error", lambda result: result.error_stats.max),
    ("average error", lambda result: result.error_stats.average),
    ("median error", lambda result: result.error_stats.median),
    ("interior max error", lambda result: result.inner_error_stats.max),
    ("interior average error", lambda result: result.inner_error_stats.average),
    ("interior median error", lambda result: result.inner_error_stats.median),
]:
    plt.figure(err_str)
    for poly_deg, color in zip(poly_degs, TABLEAU_COLORS.keys()):
        my_res = [result for result in results if result.poly_deg == poly_deg]
        hs = [1/sqrt(result.n) for result in my_res]
        errs = list(map(err_func, my_res))
        fit = linregress(np.log(hs), np.log(errs))
        plt.loglog(hs, errs, ".", color=color)
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"deg={poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
    plt.legend()
    plt.ylabel(err_str)
    plt.xlabel("$N^{-1/2}$")
    plt.savefig("media/hex_convergence_" + err_str + ".png")
