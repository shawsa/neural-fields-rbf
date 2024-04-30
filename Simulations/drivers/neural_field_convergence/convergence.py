from dataclasses import dataclass
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists, join
import pickle
from tqdm import tqdm

from manufactured import ManufacturedSolution
from neural_fields.scattered import (
    NeuralFieldSparse,
    FlatTorrusDistance,
)
from neural_fields.kernels import Gaussian
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5
from rbf.points import UnitSquare
from rbf.points.utils import get_stencil_size
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS


@dataclass
class Result:
    solver: str
    delta_t: float
    N: float
    rbf: str
    poly_deg: int
    stencil_size: int
    max_relative_error: float


if __name__ == "__main__":
    DATA_DIR = "data"
    SAVE_DATA = True

    width = 50
    t0, tf = 0, 0.1

    threshold = 0.5
    gain = 5
    weight_kernel_sd = 1
    sol_sd = 0.5
    path_radius = 10
    epsilon = 0.1

    sol = ManufacturedSolution(
        weight_kernel_sd=weight_kernel_sd,
        threshold=threshold,
        gain=gain,
        solution_sd=sol_sd,
        path_radius=path_radius,
        epsilon=epsilon,
    )

    rbf = PHS(3)
    solver = AB5(seed=RK4(), seed_steps_per_step=2)
    time_step_sizes = [1e-3, 1e-4]
    # Ns = [8000 * 2**i for i in range(3)]
    Ns = np.logspace(3.9, 4.7, 11, dtype=int)
    repeats = 5
    poly_degs = [1, 2, 3, 4]

    results = []

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
            stencil_size = get_stencil_size(deg=poly_deg)
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
                        position=3,
                        leave=False,
                    )
                ):
                    my_exact = sol.exact(*points.T, t)
                    err = np.max(np.abs(u - my_exact) / my_exact)
                    max_err = max(max_err, err)

                results.append(
                    Result(
                        solver=str(solver),
                        delta_t=delta_t,
                        N=N,
                        rbf=str(rbf),
                        poly_deg=poly_deg,
                        stencil_size=stencil_size,
                        max_relative_error=max_err,
                    )
                )

    if SAVE_DATA:

        def path(index: int):
            return join(DATA_DIR, f"convergence_data{index}.pickle")

        index = 0
        while exists(path(index)):
            index += 1

        with open(path(index), "wb") as f:
            pickle.dump(results, f)

    for delta_t in time_step_sizes:
        plt.figure(f"Convergence {delta_t=}")
        for deg in set([res.poly_deg for res in results]):
            my_res = [
                res
                for res in results
                if res.poly_deg == deg and res.delta_t == delta_t
            ]
            my_hs = [1 / np.sqrt(res.N) for res in my_res]
            my_errs = [res.max_relative_error for res in my_res]
            plt.loglog(my_hs, my_errs, ".", label=f"{deg=}")
        plt.xlabel("$N^{-1/2}$")
        plt.ylabel("Relative Error")
        plt.legend()
