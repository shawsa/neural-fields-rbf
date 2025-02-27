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

from manufactured import ManufacturedSolutionPeriodic

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.kernels import Gaussian
from neural_fields.scattered import (
    NeuralFieldSparse,
    FlatTorrusDistance,
)
import scipy.sparse

import pyvista as pv


@dataclass
class Result:
    N: int
    vor_circum: float
    rbf: RBF
    poly_deg: int
    stencil_size: int
    max_err: float


if __name__ == "__main__":
    DATA_FILE = "data/torrus_nf_1.pickle"
    SAVE_DATA = True

    R, r = 3, 1
    t0, tf = 0, 0.1

    threshold = 0.5
    gain = 5
    weight_kernel_sd = 0.025
    sol_sd = 1.1
    path_radius = 0.2
    epsilon = 0.1

    def cart_to_param(points):
        if points.ndim == 1:
            x, y, z = points
        else:
            x, y, z = points.T
        phis = np.arctan2(y, x)
        thetas = np.arctan2(z, np.sqrt(x**2 + y**2) - R)
        return np.array([phis, thetas]).T

    def param_dist(x, z):
        return FlatTorrusDistance(x_width=2 * np.pi, y_width=2 * np.pi)(
            cart_to_param(x), cart_to_param(z)
        )

    sol = ManufacturedSolutionPeriodic(
        weight_kernel_sd=weight_kernel_sd,
        threshold=threshold,
        gain=gain,
        solution_sd=sol_sd,
        path_radius=path_radius,
        epsilon=epsilon,
        period=2 * np.pi,
    )

    Ns = np.logspace(
        np.log10(16_000),
        np.log10(128_000),
        5,
        dtype=int,
    )

    solver = AB5(seed=RK4(), seed_steps_per_step=2)
    # solver = RK4()
    t0, tf = 0, 0.2
    time_step_sizes = [1e-4]
    # Ns = np.logspace(np.log10(1_000), np.log10(5_000), 5, dtype=int)
    # poly_degs = [1, 2, 3, 4]
    stencil_size = 24
    poly_degs = [1, 2, 3, 4]
    rbf = PHS(3)

    results = []
    if False:
        N = 20_000
        poly_deg = 4
        delta_2 = 1e-4
    for N in (tqdm_obj := tqdm(Ns[::-1], position=0, leave=True)):
        tqdm_obj.set_description(f"{N=} - generating surface...")
        torus = SpiralTorus(N, R=R, r=r)
        N = torus.N
        points = torus.points
        phis, thetas = cart_to_param(points).T
        valid_surface = False
        while not valid_surface:
            vor = LocalSurfaceVoronoi(
                torus.points,
                torus.normals,
                torus.implicit_surf,
            )
            trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
            valid_surface = trimesh.is_valid()

        for poly_deg in (
            tqdm_poly_deg := tqdm(poly_degs[::-1], position=1, leave=False)
        ):
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
            nf = NeuralFieldSparse(
                qf=qf,
                firing_rate=sol.firing,
                weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
                dist=param_dist,
                sparcity_tolerance=1e-16,
                verbose=True,
                tqdm_kwargs={"position": 2, "leave": False},
            )
            # mult by jacobian
            nf.conv_mat = nf.conv_mat @ scipy.sparse.diags(1 / (R + r * np.cos(thetas)))

            def rhs(t, u):
                return nf.rhs(t, u) + sol.rhs(phis, thetas, t)

            u0 = sol.exact(phis, thetas, 0)
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
                    my_exact = sol.exact(phis, thetas, t)
                    err = np.max(np.abs(u - my_exact) / my_exact)
                    max_err = max(max_err, err)
                    tqdm_obj_inner.set_description(f"{max_err=:.3E}")

            result = Result(
                N=N,
                vor_circum=vor.circum_radius,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                max_err=max_err,
            )
            results.append(result)
            tqdm_obj.set_description(f"{N=}, {str(rbf)}, {poly_deg=}, {max_err=:.3E}")

    if False:
        trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
        triangles = pv.PolyData(points, [(3, *f) for f in trimesh.simplices])
        plotter = pv.Plotter()
        plotter.add_mesh(
            triangles,
            show_edges=False,
            show_vertices=False,
            scalars=err,
            show_scalar_bar=True,
        )
        plotter.show()

    if SAVE_DATA:
        with open(DATA_FILE, "wb") as f:
            pickle.dump(results, f)

    if False:
        # for REPL use
        with open(DATA_FILE, "rb") as f:
            results = pickle.load(f)

    for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
        my_res = [result for result in results if result.poly_deg == poly_deg]
        ns = [result.N for result in my_res]
        hs = [1 / np.sqrt(result.N, dtype=float) for result in my_res]
        errs = [result.max_err for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        plt.figure("Surface NF")
        plt.loglog(hs, errs, ".", color=color)
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"$deg={poly_deg}$ ~ "
            + f"$\\mathcal{{O}}({fit.slope:.2f})$",
        )
        plt.legend()

    plt.title("Neural Field on Torus")
    plt.ylabel("Relative Error")
    # plt.xlabel("$h$")
    plt.xlabel("$\\sqrt{N}^{-1}$")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(NullFormatter())

    plt.savefig("media/torus_nf_convergence.png")
