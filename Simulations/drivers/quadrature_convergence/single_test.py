"""
Run convergence Tests for number of points
"""
from dataclasses import dataclass
from math import ceil
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
    covering_dist,
    HermiteBump,
    hex_grid,
    random_points,
    hex_stencil_min,
    PeriodicTile,
    poly_stencil_min,
    quad_test,
)

FILE_PREFIX = "media/single_test_"


rbf = PHS(3)
poly_degs = range(1, 4)
stencil_size_factor = 2
h_targets = np.logspace(-4, -5, 11, base=2)

bump_order = 4
bump_radius = 0.1
x0, y0 = 0.5, 0.5


func_name = "hermite_order4"

SAVE = False
PLOT_WHILE_RUNNING = True

repeats = 5

# get_points, point_set, repeats = (random_points, "min_energy_", 10)
# get_points, point_set, repeats = (hex_grid, "hex_", 1)

covering_sample_density = 801


colors = {deg: color for deg, color in zip(poly_degs, TABLEAU_COLORS.keys())}


def get_stencil_size(poly_deg: int) -> int:
    return hex_stencil_min(ceil(stencil_size_factor * poly_stencil_min(poly_deg)))


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
    error_stats: SummaryStats


bump = PeriodicTile(HermiteBump(order=bump_order, radius=bump_radius))

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)

exact = bump.integrate()
X, Y = np.meshgrid(
    np.linspace(0, 1, covering_sample_density),
    np.linspace(0, 1, covering_sample_density),
)

results = []
if PLOT_WHILE_RUNNING:
    plt.figure("Running")
    for deg, color in colors.items():
        plt.loglog([], [], ".", color=color, label=f"{deg=}")
    plt.legend()
    plt.ylabel("Relative Error")
    plt.xlabel("$h_{max}$")
for _ in tqdm(range(repeats), leave=False, position=0, unit="trial"):
    for h_target in (hs_prog := tqdm(h_targets[::-1], position=1)):
        points = random_points(
            hex_limit_density(h_target),
            verbose=True,
            tqdm_kwargs={"leave": False, "position": 2},
        )
        n = len(points)
        hs_prog.set_description(f"{n=}")
        for poly_deg in (deg_prog := tqdm(poly_degs[::-1], leave=False, position=4)):
            stencil_size = get_stencil_size(poly_deg)
            deg_prog.set_description(f"{poly_deg=}, k={stencil_size}")
            qf = LocalQuad(
                points,
                rbf,
                poly_deg,
                stencil_size,
                verbose=True,
                tqdm_kwargs={
                    "position": 5,
                    "leave": False,
                },
            )
            covering = covering_dist(qf, X, Y)
            errors = np.abs(
                quad_test(
                    qf,
                    bump,
                    np.array([x0]),
                    np.array([y0]),
                    verbose=False,
                )
            )
            result = Result(
                n=n,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                covering_stats=summary_stats(covering),
                error_stats=summary_stats(errors),
            )
            results.append(result)
            hs_prog.set_description(
                f"{n=}, h={result.covering_stats.max:.3E}, error={result.error_stats.max:.3E}"
            )
            if PLOT_WHILE_RUNNING:
                plt.plot(
                    result.covering_stats.max,
                    result.error_stats.max,
                    ".",
                    color=colors[poly_deg],
                )
                plt.pause(1e-3)


plt.figure()
for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    hs = [result.n**-0.5 for result in my_res]
    # hs = [result.covering_stats.max for result in my_res]
    # hs = [result.covering_stats.average for result in my_res]
    errs = [result.error_stats.max for result in my_res]
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
plt.ylabel("Relative Error")
# plt.xlabel("$h_{ave}$")
plt.xlabel("$N^{-1/2}$")
if SAVE:
    plt.savefig(FILE_PREFIX + func_name + ".png")
