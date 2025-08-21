import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.gridspec as gs
import numpy as np
import json
from scipy.spatial import Delaunay
from scipy.stats import linregress
import sympy as sym
from tqdm import tqdm
from types import SimpleNamespace

from rbf.geometry import delaunay_covering_radius_stats
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from utils import hex_stencil_min, hex_grid

from flat_quad_space_figure.utils import Gaussian


DATA_FILE = "data/flat_hex_quad_convergence.json"

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}

x, y = sym.symbols("x y", real=True)
test_funcs_symbolic = [
    1 + 0 * x,
    x**3 - y**4,
]
test_functions = [
    {
        "desc": str(expr),
        "foo": sym.lambdify((x, y), expr),
        "approx_quad": float(sym.integrate(sym.integrate(expr, (x, 0, 1)), (y, 0, 1))),
    }
    for expr in test_funcs_symbolic
]
for text_func_dict in test_functions:
    if text_func_dict["desc"] == "1":
        text_func_dict["foo"] = lambda x, y: np.ones_like(x)
        break
gauss = Gaussian(0.1)
center = (0.5, 0.5)
test_functions.append(
    {
        "desc": "Gaussian",
        "foo": lambda x, y: np.exp(
            -((x - center[0]) ** 2 + (y - center[1]) ** 2) / gauss.sd**2
        ),
        "approx_quad": np.pi * gauss.sd**2,
    }
)


rbf = PHS(3)
poly_degs = list(range(1, 5))
stencil_size = hex_stencil_min(30)
print(f"{stencil_size=}")
# Ns = list(map(int, np.logspace(np.log10(1_000), np.log10(2_000), 3)))
Ns = [200*2**i for i in range(5)]
errors = []
for _ in (tqdm_obj_status := tqdm([None], position=0, leave=True)):
    for N in (tqdm_N := tqdm(Ns[::-1], position=1, leave=False)):
        tqdm_N.set_description(f"{N=}")
        points = hex_grid(N)
        N = len(points)
        mesh = Delaunay(points)
        h, _ = delaunay_covering_radius_stats(mesh)
        for poly_deg in (tqdm_poly_deg := tqdm(poly_degs, position=2, leave=False)):
            tqdm_poly_deg.set_description(f"{poly_deg=}")
            quad = LocalQuad(
                points=points,
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
            for test_func_dict in test_functions:
                approx = quad.weights @ test_func_dict["foo"](*points.T)
                error = approx - test_func_dict["approx_quad"]
                relative_error = abs(error / test_func_dict["approx_quad"])
                tqdm_obj_status.set_description(
                    f"func={test_func_dict['desc']}: {poly_deg=} {error=:.3g}"
                )
                errors.append(
                    {
                        "func": test_func_dict["desc"],
                        "rbf": str(rbf),
                        "poly_deg": poly_deg,
                        "stencil_size": stencil_size,
                        "N": N,
                        "h": h,
                        "approx_quad": approx,
                        "error": error,
                        "relative_error": relative_error,
                        "num_negative": len(quad.weights[quad.weights < 0])
                    }
                )

# save results
with open(DATA_FILE, "w") as f:
    json.dump(errors, f)

# load results
with open(DATA_FILE, "r") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

# plot results
poly_degs = sorted(set([res.poly_deg for res in results]))
funcs = set([res.func for res in results])
figsize = (8, 7)
fig = plt.figure("Sphere Quadrature Convergence", figsize=figsize)
grid = gs.GridSpec(1, len(test_functions))
axes = [fig.add_subplot(grid[0, col]) for col, _ in enumerate(funcs)]
for ax, func in zip(axes, funcs):
    colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
    for poly_deg in poly_degs:
        color = colors[poly_deg]
        my_res = [
            result
            for result in results
            if result.poly_deg == poly_deg and result.func == func
        ]
        ns = [result.N for result in my_res]
        # hs = [1 / np.sqrt(result.N) for result in my_res]
        hs = [res.h for res in my_res]
        errs = [np.abs(result.relative_error) for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        ax.loglog(hs, errs, ".", color=color)
        ax.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
    ax.legend()
    ax.set_title(func)
    ax.set_ylabel("Relative Error")
    ax.set_xlabel("${N}^{-1/2}$")
    # ax.set_xlabel("$h$")
