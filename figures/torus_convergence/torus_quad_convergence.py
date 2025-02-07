from dataclasses import dataclass
import pickle
import numpy as np
import math
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import linregress
import sympy as sym
from sympy.abc import x, y, z
from tqdm import tqdm
from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus

from rbf.rbf import RBF, PHS
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
    rbf: RBF
    poly_deg: int
    stencil_size: int
    expected_order: int
    test_func: TestFunc
    approx: float
    error: float


if __name__ == "__main__":
    DATA_FILE = "data/torrus_quad_1.pickle"
    SAVE_DATA = True

    R, r = 3, 1
    exact = 4 * np.pi**2 * R * r

    test_func = TestFunc(1 + sym.sin(7 * x))
    # test_func = TestFunc(
    #     1
    #     + sym.exp(-((x - 2) ** 2) + y**2 + z**2)
    #     - sym.exp(-((x + 2) ** 2) + y**2 + z**2)
    # )

    Ns = np.logspace(
        np.log10(8_000),
        np.log10(16_000),
        2,
        dtype=int,
    )

    target_orders = [4, 6, 8, 10]

    results = []
    for N in (tqdm_N := tqdm(Ns[::-1], position=1, leave=True)):
        tqdm_N.set_description(f"{N=} - generating surface...")
        torus = SpiralTorus(N, R=R, r=r)
        N = torus.N
        points = torus.points
        valid_surface = False
        while not valid_surface:
            vor = LocalSurfaceVoronoi(
                torus.points,
                torus.normals,
                torus.implicit_surf,
            )
            trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
            valid_surface = trimesh.is_valid()

        for target_order in (
            tqdm_poly_deg := tqdm(target_orders, position=2, leave=False)
        ):
            tqdm_poly_deg.set_description(f"{target_order=}")
            interp_order = target_order + 2
            rbf = PHS(interp_order)
            poly_deg = (interp_order - 2) // 2
            stencil_size = max(12, math.ceil(1.2 * (1 + poly_deg) * poly_deg // 2))

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
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                expected_order=target_order,
                approx=approx,
                error=error,
                test_func=str(test_func),
            )
            results.append(result)
            tqdm_N.set_description(f"{N=}, {str(rbf)}, {poly_deg=}, {error=:.3E}")

    if SAVE_DATA:
        with open(DATA_FILE, "wb") as f:
            pickle.dump(results, f)

    if False:
        # for REPL use
        with open(DATA_FILE, "rb") as f:
            results = pickle.load(f)

    my_res = [result for result in results if result.test_func == str(test_func)]
    for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
        my_res = [
            result
            for result in results
            if result.test_func == str(test_func) and result.poly_deg == poly_deg
        ]
        hs = [result.h for result in my_res]
        ns = [result.N for result in my_res]
        errs = [result.error for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        plt.figure(f"{test_func=}")
        plt.loglog(hs, errs, ".", color=color)
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
        plt.legend()

    plt.title(f"f={test_func}")
    plt.ylabel("Relative Error")
    plt.xlabel("$h$")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(NullFormatter())

    plt.savefig(f"media/torus_{test_func}.png")
