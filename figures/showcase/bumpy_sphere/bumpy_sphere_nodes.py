import numpy as np
import numpy.linalg as la
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.sphere import SpherePoints
from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pickle

from bumpy_sphere_utils import approximate_normal_vectors, sphere_to_bumpy_sphere

N = 64_000
num_bumps = 100
bump_amplitude = 0.1
bump_sd = 0.1

np.random.seed(0)
bump_centers = SpherePoints(num_bumps, auto_settle=False, verbose=True)
bump_centers.jostle(50, verbose=True)
bump_centers.settle(0.01, repeat=1_00, verbose=True)

np.random.seed(0)
sphere = SpherePoints(N, auto_settle=False)
sphere.points[-num_bumps:] = bump_centers.points.copy()
sphere.num_fixed = num_bumps
sphere.num_mutable = N - num_bumps
sphere.jostle(5_000, verbose=True)
sphere.settle(0.01, repeat=1_00, verbose=True)

vor = LocalSurfaceVoronoi(
    sphere.points,
    sphere.points,
    lambda x: la.norm(x, axis=1) - 1,
)

points = sphere_to_bumpy_sphere(
    sphere.points,
    bump_centers=bump_centers.points,
    bump_amplitude=bump_amplitude,
    bump_sd=bump_sd,
)

normals = approximate_normal_vectors(points, num_neighbors=50)
trimesh = TriMesh(points, vor.triangles, normals=normals)
quad = SurfaceQuad(
    trimesh=trimesh,
    rbf=PHS(3),
    poly_deg=3,
    stencil_size=32,
    verbose=True,
    tqdm_kwargs={
        "position": 0,
        "leave": False,
        "desc": "Calculating weights",
    },
)

np.savetxt("bump_centers.csv", bump_centers.points)

with open(f"data/bumpy_sphere_{N}_quad.pickle", "wb") as f:
    pickle.dump(
        (
            {
                "num_bumps": num_bumps,
                "bump_amplitude": bump_amplitude,
                "bump_sd": bump_sd,
            },
            quad,
        ),
        f,
    )

# plotter = pv.Plotter()
# plotter.add_mesh(
#     pv.PolyData(points, [(3, *f) for f in vor.triangles]),
#     show_edges=False,
# )
# plotter.show()

normal_vecs = pv.PolyData(points)
normal_vecs["vectors"] = 0.01 * normals
normal_vecs.set_active_vectors("vectors")
plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(points, [(3, *f) for f in trimesh.simplices]),
    show_edges=False,
    scalars=quad.weights,
)
plotter.add_mesh(normal_vecs.arrows, color="red", show_scalar_bar=False)
plotter.show()
