from dataclasses import dataclass, asdict
from itertools import product
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
from scipy.spatial import Delaunay
from tqdm import tqdm

from manufactured import ManufacturedSolutionPeriodic
from neural_fields.scattered import (
    NeuralFieldSparse,
    FlatTorrusDistance,
)
from neural_fields.kernels import Gaussian
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5
from rbf.geometry import delaunay_covering_radius_stats
from rbf.points import UnitSquare
# from rbf.points.utils import get_stencil_size
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS

from scipy.stats import linregress


@dataclass
class Result:
    N: int
    h: float
    delta_t: float
    rbf: str
    poly_deg: int
    stencil_size: int
    max_relative_error: float
    max_err: float


if __name__ == "__main__":
    DATA_FILE = "data/flat_nf_convergence.json"
    SAVE_DATA = True

    width = 2 * np.pi
    t0, tf = 0, 0.1

    threshold = 0.5
    gain = 5
    weight_kernel_sd = 0.025
    sol_sd = 1.1
    path_radius = 0.2
    epsilon = 0.1

    sol = ManufacturedSolutionPeriodic(
        weight_kernel_sd=weight_kernel_sd,
        threshold=threshold,
        gain=gain,
        solution_sd=sol_sd,
        path_radius=path_radius,
        epsilon=epsilon,
        period=width,
    )

    rbf = PHS(3)
    solver = AB5(seed=RK4(), seed_steps_per_step=2)
    delta_t = 1e-5
    Ns = np.logspace(np.log10(32_000), np.log10(64_000), 5, dtype=int)
    repeats = 1
    poly_degs = [1, 2, 3, 4]
    stencil_size = 21

    results = []
    np.random.seed(0)
    for repeat in tqdm(range(repeats), desc="repeat"):
        for N, poly_deg in (
            tqdm_obj := tqdm(
                list(product(reversed(Ns), reversed(poly_degs))),
                position=1,
                leave=False,
            )
        ):
            tqdm_obj.set_description(f"{N=}, {poly_deg=} generating points")
            points = UnitSquare(N).points * width - width / 2
            mesh = Delaunay(points)
            h, _ = delaunay_covering_radius_stats(mesh)
            # stencil_size = get_stencil_size(deg=poly_deg, stability_factor=1.5)
            tqdm_obj.set_description(f"{N=}, {poly_deg=} constructing stencil")
            qf = LocalQuad(
                points=points,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                verbose=True,
                tqdm_kwargs={"position": 2, "leave": False},
            )
            tqdm_obj.set_description(f"{N=}, {poly_deg=} constructing conv_mat")
            nf = NeuralFieldSparse(
                qf=qf,
                firing_rate=sol.firing,
                weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
                dist=FlatTorrusDistance(x_width=width, y_width=width),
                sparcity_tolerance=1e-16,
                verbose=True,
                tqdm_kwargs={"position": 2, "leave": False},
            )

            def rhs(t, u):
                return nf.rhs(t, u) + sol.rhs(*points.T, t)

            u0 = sol.exact(*points.T, 0)
            time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
            max_rel_err = 0
            max_err = 0
            for t, u in (
                tqdm_obj_inner := tqdm(
                    zip(time.array, solver.solution_generator(u0, rhs, time)),
                    total=len(time.array),
                    position=2,
                    leave=False,
                )
            ):
                my_exact = sol.exact(*points.T, t)
                err = np.max(np.abs(u - my_exact))
                rel_err = np.max(np.abs(u - my_exact) / my_exact)
                max_err = max(max_err, err)
                max_rel_err = max(max_err, rel_err)
                tqdm_obj_inner.set_description(f"err={max_err:.3E}")

            results.append(
                Result(
                    N=int(N),
                    h=h,
                    delta_t=delta_t,
                    rbf=str(rbf),
                    poly_deg=poly_deg,
                    stencil_size=stencil_size,
                    max_relative_error=max_rel_err,
                    max_err=max_err,
                )
            )

    if SAVE_DATA:
        results_dicts = [asdict(result) for result in results]
        with open(DATA_FILE, "w") as f:
            json.dump(results_dicts, f)

    if False:
        # for REPL use
        with open(DATA_FILE, "r") as f:
            results = json.load(f)
        results = [Result(**result) for result in results]

    colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
    plt.figure(f"Convergence {delta_t=}")
    for deg in set([res.poly_deg for res in results]):
        my_res = [
            res for res in results if res.poly_deg == deg and res.delta_t == delta_t
        ]
        my_hs = [res.h for res in my_res]
        my_errs = [res.max_relative_error for res in my_res]
        fit = linregress(np.log(my_hs), np.log(my_errs))
        plt.loglog(my_hs, my_errs, ".", color=colors[deg], label=f"{deg=}")
        plt.loglog(
            my_hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in my_hs],
            "-",
            color=colors[deg],
            label=f"$\\mathcal{{O}}({fit.slope:.2f})$",
        )
    plt.xlabel("$h$")
    plt.ylabel("Relative Error")
    plt.legend()
