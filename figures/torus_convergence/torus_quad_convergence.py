from dataclasses import dataclass, asdict
import json
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import linregress
import sympy as sym
from sympy.abc import x, y, z
from tqdm import tqdm
from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus

from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad


class TestFunc:
    def __init__(self, expr):
        self.expr = expr
        self.foo = sym.lambdify((x, y, z), expr)

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        if type(self.expr) in [int, float]:
            return np.ones(len(points))
        return self.foo(*points.T)

    def __repr__(self) -> str:
        return str(self.expr)


@dataclass
class Result:
    N: int
    h: float
    rbf: str
    poly_deg: int
    stencil_size: int
    test_func: TestFunc
    approx: float
    error: float


if __name__ == "__main__":
    DATA_FILE = "data/torus_quad.json"
    SAVE_DATA = False

    R, r = 3, 1
    exact = 4 * np.pi**2 * R * r

    test_func = TestFunc(1 + sym.sin(7 * x))
    # test_func = TestFunc(
    #     1
    #     + sym.exp(-((x - 2) ** 2) + y**2 + z**2)
    #     - sym.exp(-((x + 2) ** 2) + y**2 + z**2)
    # )

    Ns = np.logspace(
        np.log10(32_000),
        np.log10(64_000),
        11,
        dtype=int,
    )
    rbf = PHS(3)
    poly_degs = [1, 2, 3, 4]
    stencil_size = 40

    results = []
    for _ in (tqdm_status := tqdm([None], position=0, leave=True)):
        tqdm_status.set_description("Calculating first quadrature")
        for N in (tqdm_N := tqdm(Ns[::-1], position=1, leave=True)):
            tqdm_N.set_description(f"{N=} - generating surface")
            torus = SpiralTorus(N, R=R, r=r)
            N = torus.N
            points = torus.points
            valid_surface = False
            tqdm_N.set_description(f"{N=} - constructing mesh")
            while not valid_surface:
                vor = LocalSurfaceVoronoi(
                    torus.points,
                    torus.normals,
                    torus.implicit_surf,
                )
                trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
                valid_surface = trimesh.is_valid()

            for poly_deg in (
                    tqdm_poly_deg := tqdm(poly_degs[::-1], position=2, leave=False)
            ):
                tqdm_poly_deg.set_description(f"{poly_deg=}")

                quad = SurfaceQuad(
                    trimesh=trimesh,
                    rbf=rbf,
                    poly_deg=poly_deg,
                    stencil_size=stencil_size,
                    verbose=True,
                    tqdm_kwargs={
                        "position": 3,
                        "leave": False,
                        "desc": "Calculating weights",
                    },
                )

                approx = quad.weights @ test_func(points)
                error = abs(approx - exact) / exact

                result = Result(
                    N=N,
                    h=vor.circum_radius,
                    rbf=str(rbf),
                    poly_deg=poly_deg,
                    stencil_size=stencil_size,
                    approx=approx,
                    error=error,
                    test_func=str(test_func),
                )
                results.append(result)
                tqdm_status.set_description(f"{N=}, {str(rbf)}, {poly_deg=}, {error=:.3E}")

    if SAVE_DATA:
        results_dicts = [asdict(result) for result in results]
        with open(DATA_FILE, "w") as f:
            json.dump(results_dicts, f)

    if False:
        # for REPL use
        with open(DATA_FILE, "r") as f:
            results = json.load(f)
        results = [Result(**result) for result in results]

    my_res = [result for result in results if result.test_func == str(test_func)]
    for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
        my_res = [
            result
            for result in results
            if result.test_func == str(test_func)
            and result.poly_deg == poly_deg
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
            label=f"deg={poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
        plt.legend()

    plt.title(f"f={test_func}")
    plt.ylabel("Relative Error")
    # plt.xlabel("$h$")
    plt.xlabel("$\\sqrt{N}^{-1}$")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.show()
