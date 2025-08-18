import json
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.gridspec as gs
import numpy as np
import numpy.linalg as la
import pickle

import pyvista as pv
from scipy.stats import linregress
from scipy.spatial import ConvexHull
from tqdm import tqdm
from types import SimpleNamespace


from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

SPHERE_BEST_QUAD_FILE = "data/sphere_best_quad.pickle"
SPHERE_CONVERGENCE_DATA_FILE = "data/sphere_convergence.json"

USE_HOPS = True


def sphere_base():
    """
    Return twelve points on the unit sphere that are vertices
    of a regular icosahedron.
    """
    phi = (1 + np.sqrt(5)) / 2
    points = np.zeros((12, 3), dtype=float)
    points[0] = [0, 1, phi]
    points[1] = [0, -1, phi]
    points[2] = [0, 1, -phi]
    points[3] = [0, -1, -phi]
    points[4] = [1, phi, 0]
    points[5] = [-1, phi, 0]
    points[6] = [1, -phi, 0]
    points[7] = [-1, -phi, 0]
    points[8] = [phi, 0, 1]
    points[9] = [-phi, 0, 1]
    points[10] = [phi, 0, -1]
    points[11] = [-phi, 0, -1]
    points /= la.norm(points, axis=1)[:, np.newaxis]
    hull = ConvexHull(points)
    return pv.PolyData(points, [(3, *f) for f in hull.simplices])


def sphere_refine(mesh: pv.PolyData, num_refinements: int) -> pv.PolyData:
    mesh = mesh.subdivide(num_refinements)
    mesh.points /= la.norm(mesh.points, axis=1)[:, np.newaxis]
    return mesh


mesh = sphere_refine(sphere_base(), 4)

# plotter = pv.Plotter()
# plotter.add_mesh(
#     mesh,
#     show_edges=True,
# )
# plotter.show()

test_functions = [
    {
        "desc": "1",
        "foo": lambda x: np.ones_like(x[:, 0]),
        "approx_quad": 4*np.pi,
    },
    {
        "desc": "trig",
        "foo": lambda x: np.sin(x[:, 0]) * np.cos(2 * x[:, 1]) * np.cos(3 * x[:, 2])
        + 1,
        "approx_quad": 4*np.pi,
    },
    {
        "desc": "poly",
        "foo": lambda x: x[:, 0] ** 3 * x[:, 1] ** 2 * x[:, 2] ** 4 + 5,
    },
]


mesh = sphere_refine(sphere_base(), 8)

trimesh = TriMesh(mesh.points, mesh.faces.reshape((-1, 4))[:, 1:], normals=mesh.points)

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

with open(SPHERE_BEST_QUAD_FILE, "wb") as f:
    pickle.dump({"trimesh": trimesh, "qf": qf}, f)

if False:
    with open(SPHERE_BEST_QUAD_FILE, "rb") as f:
        best_dict = pickle.load(f)
    shape = best_dict["shape"]
    vor = best_dict["vor"]
    trimesh = best_dict["shape"]
    qf = best_dict["qf"]

for test_func_dict in test_functions:
    if "approx_quad" not in test_func_dict.keys():
        test_func_dict["approx_quad"] = qf.weights @ test_func_dict["foo"](qf.points)


base_refinements = 3
final_refinement = 8
rbf = PHS(3)
poly_degs_and_stencils = [
    (1, 12),
    (2, 12),
    (3, 12),
    (4, 15),
]

errors = []
mesh = sphere_refine(sphere_base(), base_refinements - 1)
for _ in (tqdm_obj_status := tqdm([None], position=0, leave=True)):
    for refinement_level in (
        tqdm_obj_Ns := tqdm(
            range(base_refinements, final_refinement + 1), position=1, leave=True
        )
    ):
        mesh = sphere_refine(mesh, 1)
        N = len(mesh.points)
        tqdm_obj_Ns.set_description(f"{N=: 6d}")
        for poly_deg, stencil_size in (
            tqdm_obj_poly_deg := tqdm(poly_degs_and_stencils, position=2, leave=False)
        ):
            tqdm_obj_poly_deg.set_description(f"{poly_deg=}")
            trimesh = TriMesh(
                mesh.points, mesh.faces.reshape((-1, 4))[:, 1:], normals=mesh.points
            )
            quad = SurfaceQuad(
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
                approx = quad.weights @ test_func_dict["foo"](quad.points)
                error = approx - test_func_dict["approx_quad"]
                relative_error = error / test_func_dict["approx_quad"]
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
                        "approx_quad": approx,
                        "error": error,
                        "relative_error": relative_error,
                    }
                )

# save results
with open(SPHERE_CONVERGENCE_DATA_FILE, "w") as f:
    json.dump(errors, f)

# load results
with open(SPHERE_CONVERGENCE_DATA_FILE, "r") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = sorted(set([res.poly_deg for res in results]))
funcs = set([res.func for res in results])
# plot results
figsize = (8, 7)
fig = plt.figure("Sphere Quadrature Convergence", figsize=figsize)

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
    # ax.set_xlabel("$h$")

if False:
    mesh = sphere_base()
    while len(mesh.points) < len(quad.weights):
        mesh = sphere_refine(mesh, 1)

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        show_edges=False,
        scalars=quad.weights,
        cmap="viridis",
    )
    plotter.show()

    mesh = sphere_base()
    my_points = mesh.points
    mesh = sphere_refine(mesh, 2)

    plotter = pv.Plotter()
    plotter.add_points(
        my_points,
        color="green",
        point_size=20,
        render_points_as_spheres=True,
    )
    plotter.add_mesh(
        mesh,
        show_edges=True,
    )
    plotter.show()
