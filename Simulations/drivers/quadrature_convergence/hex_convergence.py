"""
Run convergence Tests for number of points
"""
from collections import namedtuple
from dataclasses import dataclass
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
from utils import (
    HermiteBump,
    PeriodicTile,
    hex_grid,
    hex_stencil_min,
    covering_dist,
    quad_test,
)


DATA_FILE = "data/hex_convergence_hermite.pickle"

rbf = PHS(3)
poly_degs = range(1, 6)
stencil_size_factor = 1.5
h_targets = np.logspace(-4, -7, 11, base=2)[::-1]

sample_density = 101
bump_order = 3
bump_radius = 0.1

inner_dist = 0.2


def get_stencil_size(poly_deg: int) -> int:
    return hex_stencil_min(ceil(stencil_size_factor * (poly_deg + 1) * (poly_deg + 2)))


@dataclass
class SummaryStats:
    min: float
    max: float
    median: float
    average: float


def summary_stats(vec: np.ndarray[float]) -> SummaryStats:
    return SummaryStats(
        min=np.min(vec),
        max=np.max(vec),
        median=np.median(vec),
        average=np.average(vec),
    )


@dataclass
class Result:
    n: int
    poly_deg: int
    stencil_size: int
    covering_stats: SummaryStats
    inner_covering_stats: SummaryStats
    error_stats: SummaryStats
    inner_error_stats: SummaryStats


bump = PeriodicTile(HermiteBump(order=bump_order, radius=bump_radius))

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

exact = bump.integrate()
X, Y = np.meshgrid(np.linspace(0, 1, sample_density), np.linspace(0, 1, sample_density))
inner = np.logical_and(X > inner_dist, X < 1 - inner_dist)
inner = np.logical_and(inner, Y > inner_dist)
inner = np.logical_and(inner, Y < 1 - inner_dist)


results = []
for h_target in (hs_prog := tqdm(h_targets, position=0)):
    points = hex_grid(hex_limit_density(h_target))
    n = len(points)
    hs_prog.set_description(f"{n=}")
    for poly_deg in (deg_prog := tqdm(poly_degs, leave=False, position=1)):
        deg_prog.set_description(f"{poly_deg=}")
        stencil_size = get_stencil_size(poly_deg)
        qf = LocalQuad(points, rbf, poly_deg, stencil_size)
        covering = covering_dist(qf, X, Y)
        errors = np.abs(
            quad_test(
                qf,
                bump,
                X,
                Y,
                verbose=True,
                tqdm_kwargs={"leave": False, "position": 2},
            )
        )
        result = Result(
            n=n,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
            covering_stats=summary_stats(covering),
            inner_covering_stats=summary_stats(covering[inner]),
            error_stats=summary_stats(errors),
            inner_error_stats=summary_stats(errors[inner]),
        )
        results.append(result)
        hs_prog.set_description(
            f"{n=}, h={result.covering_stats.max:.3E}, error={result.error_stats.max:.3E}"
        )


HexConvergenceData = namedtuple(
    "HexConvergenceData",
    [
        "bump_order",
        "bump_radius",
        "inner_distance",
        "sample_density",
        "rbf",
        "results",
    ],
)

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
        # hs = [result.covering_stats.average for result in my_res]
        hs = [result.n**-2 for result in my_res]
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
    plt.xlabel("$h_{max}$")
    plt.savefig("media/hex_convergence_" + err_str + ".png")
