from dataclasses import dataclass
import pickle
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

from rbf.rbf import RBF, PHS
from rbf.surface import TriMesh, SurfaceQuad

DATA_FILE = "data/torrus1.pickle"
SAVE_DATA = True


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
    test_func: TestFunc
    approx: float
    error: float


R, r = 3, 1
exact = 4 * np.pi**2 * R * r
test_functions = list(
    map(
        TestFunc,
        [
            1,
            1 + y,
            1 + z,
            1 + x * y,
            1 + sym.sin(7 * x),
        ],
    )
)

rbf = PHS(3)
stencil_size = 24
poly_degs = [0, 1, 2, 3]

repeats = 1
Ns = np.logspace(
    np.log10(4_000),
    np.log10(64_000),
    21,
    dtype=int,
)

# repeats = 1
# Ns = np.logspace(2, 2.5, 3, dtype=int)

results = []
for trial in (tqdm_trial := tqdm(range(repeats), position=1, leave=True)):
    for N in (tqdm_N := tqdm(Ns[::-1], position=2, leave=False)):
        tqdm_N.set_description(f"{N=}")
        for poly_deg in (tqdm_poly_deg := tqdm(poly_degs, position=3, leave=False)):
            tqdm_poly_deg.set_description(f"{poly_deg=}")
            torus = SpiralTorus(N, R=R, r=r)
            N = torus.N
            points = torus.points
            valid_surface = False
            if not valid_surface:
                vor = LocalSurfaceVoronoi(
                    torus.points,
                    torus.normals,
                    torus.implicit_surf,
                    verbose=True,
                    tqdm_args={
                        "position": 4,
                        "leave": False,
                    },
                )
                trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
                valid_surface = trimesh.is_valid()

            quad = SurfaceQuad(
                trimesh=trimesh,
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

            for test_func in test_functions:
                approx = quad.weights @ test_func(points)
                error = abs(approx - exact) / exact

                result = Result(
                    N=N,
                    h=vor.circum_radius,
                    rbf=rbf,
                    poly_deg=poly_deg,
                    stencil_size=stencil_size,
                    approx=approx,
                    error=error,
                    test_func=str(test_func),
                )
                results.append(result)
                tqdm_trial.set_description(
                    f"f={test_func}, {trial=}, {N=}, {poly_deg=}, {error=:.3E}"
                )

if SAVE_DATA:
    with open(DATA_FILE, "wb") as f:
        pickle.dump(results, f)

if False:
    # for REPL use
    with open(DATA_FILE, "rb") as f:
        results = pickle.load(f)

for test_func in test_functions:
    my_res = [result for result in results if result.test_func == str(test_func)]
    hs = [result.N**-0.5 for result in my_res]
    ns = [result.N for result in my_res]
    for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
        errs = [result.error for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        plt.figure(f"{test_func=}")
        plt.loglog(hs, errs, "k.")
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "k-",
            label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
        plt.legend()
        plt.loglog(
            [res.N**-0.5 for res in results], [res.error for res in results], "k."
        )

    plt.title(f"f={test_func}")
    plt.ylabel("Relative Error")
    plt.xlabel("$N^{-1/2}$")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(NullFormatter())
    plt.xticks()
    plt.xticks([n**-0.5 for n in Ns[::4]], [f"${n}^{{-1/2}}$" for n in Ns[::4]])

    plt.savefig(f"media/torus_{test_func}.png")
