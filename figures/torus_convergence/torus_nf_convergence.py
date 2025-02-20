from dataclasses import dataclass
import pickle
import numpy as np
import math
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import linregress

from tqdm import tqdm
from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus

from rbf.rbf import RBF, PHS
from rbf.surface import TriMesh, SurfaceQuad

from manufactured import (
    weight_kernel,
    dist,
    firing_rate,
    exact,
    forcing,
    cart_to_param,
)

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.scattered import (
    NeuralField,
)


@dataclass
class Result:
    N: int
    h: float
    rbf: RBF
    poly_deg: int
    stencil_size: int
    max_err: float


if __name__ == "__main__":
    DATA_FILE = "data/torrus_nf_1.pickle"
    SAVE_DATA = True

    R, r = 3, 1

    Ns = np.logspace(
        np.log10(1_000),
        np.log10(32_000),
        2,
        dtype=int,
    )

    solver = AB5(seed=RK4(), seed_steps_per_step=2)
    t0, tf = 0, 1
    time_step_sizes = [1e-4]
    # Ns = np.logspace(np.log10(1_000), np.log10(5_000), 5, dtype=int)
    # poly_degs = [1, 2, 3, 4]
    stencil_size = 24
    poly_degs = [2]
    rbf = PHS(3)

    results = []
    for N in (tqdm_obj := tqdm(Ns[::-1], position=0, leave=True)):
        tqdm_obj.set_description(f"{N=} - generating surface...")
        torus = SpiralTorus(N, R=R, r=r)
        N = torus.N
        points = torus.points
        phis, thetas = cart_to_param(points)
        valid_surface = False
        while not valid_surface:
            vor = LocalSurfaceVoronoi(
                torus.points,
                torus.normals,
                torus.implicit_surf,
            )
            trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
            valid_surface = trimesh.is_valid()

        for poly_deg in (tqdm_poly_deg := tqdm(poly_degs, position=1, leave=False)):
            tqdm_poly_deg.set_description(f"{poly_deg=}")
            # rbf = PHS(max(2, 2 * poly_deg - 2))
            # stencil_size = max(
            #     12, math.ceil(1.5 * (2 + poly_deg) * (1 + poly_deg) // 2)
            # )

            tqdm_obj.set_description(f"{N=}, {poly_deg=} constructing stencil")
            qf = SurfaceQuad(
                trimesh=trimesh,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                verbose=True,
                tqdm_kwargs={
                    "position": 2,
                    "leave": False,
                    "desc": "Calculating weights",
                },
            )
            tqdm_obj.set_description(f"{N=}, {poly_deg=} constructing conv_mat")
            nf = NeuralField(
                qf=qf,
                firing_rate=firing_rate,
                weight_kernel=weight_kernel,
                dist=dist,
                verbose=True,
                tqdm_kwargs={"position": 2, "leave": False},
            )

            def rhs(t, u):
                return nf.rhs(t, u) + forcing(t, phis, thetas)

            u0 = exact(t0, phis, thetas)
            for delta_t in tqdm(time_step_sizes, position=2, leave=False):
                time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
                max_err = 0
                tqdm_obj.set_description(
                    f"{N=}, {delta_t=:.3E}, {poly_deg=} time stepping"
                )
                for t, u in (
                    tqdm_obj_inner := tqdm(
                        zip(time.array, solver.solution_generator(u0, rhs, time)),
                        total=len(time.array),
                        position=2,
                        leave=False,
                    )
                ):
                    my_exact = exact(t, phis, thetas)
                    err = np.max(np.abs(u - my_exact) / my_exact)
                    max_err = max(max_err, err)

            result = Result(
                N=N,
                h=vor.circum_radius,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                max_err=max_err,
            )
            results.append(result)
            tqdm_obj.set_description(f"{N=}, {str(rbf)}, {poly_deg=}, {max_err=:.3E}")

    if SAVE_DATA:
        with open(DATA_FILE, "wb") as f:
            pickle.dump(results, f)

    if False:
        # for REPL use
        with open(DATA_FILE, "rb") as f:
            results = pickle.load(f)

    my_res = [result for result in results if result.test_func == str(test_func)]
    for expected_order, color in zip(target_orders, TABLEAU_COLORS):
        my_res = [
            result
            for result in results
            if result.test_func == str(test_func)
            and result.expected_order == expected_order
        ]
        hs = [result.h for result in my_res]
        ns = [result.N for result in my_res]
        sqrt_ns = [np.sqrt(result.N, dtype=float) for result in my_res]
        errs = [result.error for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        plt.figure(f"{test_func=}")
        plt.loglog(hs, errs, ".", color=color)
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"Expected: $\\mathcal{{O}}({expected_order})$ ~ "
            + f"Measured: $\\mathcal{{O}}({fit.slope:.2f})$",
        )
        plt.legend()

    plt.title(f"f={test_func}")
    plt.ylabel("Relative Error")
    # plt.xlabel("$h$")
    plt.xlabel("$\\sqrt{N}$")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(NullFormatter())

    plt.savefig(f"media/torus_{test_func}.png")
