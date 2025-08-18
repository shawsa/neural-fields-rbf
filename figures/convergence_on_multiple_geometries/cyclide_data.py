import json
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.gridspec as gs
import numpy as np
import pickle

import pyvista as pv
from scipy.stats import linregress
from tqdm import tqdm
from types import SimpleNamespace


from min_energy_points import LocalSurfaceVoronoi
from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

from geometries import SpiralCyclide

CYCLIDE_BEST_QUAD_FILE = "data/cyclide_best_quad.pickle"
CYCLIDE_CONVERGENCE_DATA_FILE = "data/cyclide_convergence.json"

USE_HOPS = True

test_functions = [
    {
        "desc": "1",
        "foo": lambda x: np.ones_like(x[:, 0]),
    },
    {
        "desc": "trig",
        "foo": lambda x: np.sin(x[:, 0]) * np.cos(2 * x[:, 1]) * np.cos(3 * x[:, 2])
        + 1,
    },
    {
        "desc": "poly",
        "foo": lambda x: x[:, 0] ** 3 * x[:, 1] ** 2 * x[:, 2] ** 4 + 5,
    },
]

N_max = 128_000
shape = SpiralCyclide(N_max)

vor = LocalSurfaceVoronoi(
    shape.points,
    shape.normals,
    shape.implicit_surf,
    max_neighbors=15,
    verbose=True,
)
trimesh = TriMesh(shape.points, vor.triangles, normals=shape.normals)

qf = SurfaceQuad(
    trimesh=trimesh,
    rbf=PHS(3),
    poly_deg=6,
    stencil_size=30,
    hops_instead_of_distance=USE_HOPS,
    verbose=True,
    tqdm_kwargs={
        "desc": "Calculating weights",
    },
)

with open(CYCLIDE_BEST_QUAD_FILE, "wb") as f:
    pickle.dump({"shape": shape, "vor": vor, "trimesh": trimesh, "qf": qf}, f)

if False:
    with open(CYCLIDE_BEST_QUAD_FILE, "rb") as f:
        best_dict = pickle.load(f)
    shape = best_dict["shape"]
    vor = best_dict["vor"]
    trimesh = best_dict["shape"]
    qf = best_dict["qf"]

for test_func_dict in test_functions:
    test_func_dict["approx_quad"] = qf.weights @ test_func_dict["foo"](shape.points)


Ns = [2**i * 2_000 for i in range(6)]
rbf = PHS(3)
poly_degs_and_stencils = [
    (1, 12),
    (2, 12),
    (3, 12),
    (4, 15),
]

errors = []

np.random.seed(0)
for _ in (tqdm_obj_status := tqdm([None], position=0, leave=True)):
    for N in (tqdm_obj_Ns := tqdm(Ns[::-1], position=1, leave=True)):
        shape = SpiralCyclide(N)
        N = shape.N
        tqdm_obj_Ns.set_description(f"{N=: 6d}")
        for poly_deg, stencil_size in (
            tqdm_obj_poly_deg := tqdm(poly_degs_and_stencils, position=2, leave=False)
        ):
            tqdm_obj_poly_deg.set_description(f"{poly_deg=}")
            vor = LocalSurfaceVoronoi(
                shape.points,
                shape.normals,
                shape.implicit_surf,
            )
            trimesh = TriMesh(shape.points, vor.triangles, normals=shape.normals)
            qf = SurfaceQuad(
                trimesh=trimesh,
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
                hops_instead_of_distance=USE_HOPS,
                verbose=True,
                tqdm_kwargs={
                    "desc": "Calculating weights",
                    "position": 3,
                    "leave": False,
                },
            )
            for test_func_dict in test_functions:
                approx = qf.weights @ test_func_dict["foo"](shape.points)
                error = approx - test_func_dict["approx_quad"]
                relative_error = error / test_func_dict["approx_quad"]
                tqdm_obj_status.set_description(
                    f"{test_func_dict['desc']}: {poly_deg=} {error=:.3g}"
                )
                errors.append(
                    {
                        "func": test_func_dict["desc"],
                        "rbf": str(rbf),
                        "poly_deg": poly_deg,
                        "stencil_size": stencil_size,
                        "N": N,
                        "h": vor.circum_radius,
                        "approx_quad": approx,
                        "error": error,
                        "relative_error": relative_error,
                    }
                )

# save results
with open(CYCLIDE_CONVERGENCE_DATA_FILE, "w") as f:
    json.dump(errors, f)

# load results
with open(CYCLIDE_CONVERGENCE_DATA_FILE, "r") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = sorted(set([res.poly_deg for res in results]))
funcs = set([res.func for res in results])
# plot results
figsize = (8, 7)
fig = plt.figure("Cyclide Quadrature Convergence", figsize=figsize)

grid = gs.GridSpec(1, 3)
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
        hs = [1 / np.sqrt(result.N) for result in my_res]
        # hs = [res.h for res in my_res]
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
    ax.set_xlabel("$h$")

if False:
    trimesh = TriMesh(shape.points, vor.triangles, normals=shape.normals)

    plotter = pv.Plotter()
    plotter.add_mesh(
        pv.PolyData(shape.points, [(3, *f) for f in vor.triangles]),
        show_edges=True,
        # scalars=np.log10(np.abs(qf.weights)),
    )
    face_index = 200
    _, closest_verts = vor.tree.query(trimesh.face(face_index).center, 20)
    size = 20
    plotter.add_points(
        shape.points[trimesh.find_closest_by_hops(face_index, 20)],
        color="red",
        point_size=size,
    )
    plotter.add_points(
        shape.points[closest_verts] + np.c_[0, -0.005, 0],
        color="green",
        point_size=size * 0.7,
    )
    plotter.add_points(trimesh.face(face_index).center, color="black", point_size=size)
    plotter.show()
