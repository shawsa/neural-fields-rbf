import numpy as np
import numpy.linalg as la
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.sphere import SpherePoints
from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pickle
from blood_cell_utils import approximate_normal_vectors, flatten


N = 64_000
amplitudes = [0, 0.4, 0.8]
shapes = [1.7 for _ in range(len(amplitudes))]

np.random.seed(0)
sphere = SpherePoints(N, auto_settle=False, verbose=True)
sphere.jostle(5_000, verbose=True)
sphere.settle(0.01, repeat=1_00, verbose=True)

vor = LocalSurfaceVoronoi(
    sphere.points,
    sphere.points,
    lambda x: la.norm(x, axis=1) - 1,
)

for index, (amplitude, shape) in enumerate(zip(amplitudes, shapes)):
    print(f"{amplitude=}, {shape=}")

    points = flatten(sphere.points, amplitude=amplitude, shape=shape)
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

    params = {
        "amplitude": amplitude,
        "shape": shape,
    }
    with open(f"data/blood_cell_{N}_quad_index{index}.pickle", "wb") as f:
        pickle.dump(
            (
                params,
                quad,
            ),
            f,
        )

# plotter = pv.Plotter()
# plotter.add_points(pv.PolyData(sphere.points))
# plotter.show()
# 
# plotter = pv.Plotter()
# plotter.add_mesh(
#     pv.PolyData(points, [(3, *f) for f in vor.triangles]),
#     show_edges=False,
# )
# plotter.show()
# 
# normal_vecs = pv.PolyData(points)
# normal_vecs["vectors"] = 0.01 * normals
# normal_vecs.set_active_vectors("vectors")
# plotter = pv.Plotter()
# plotter.add_mesh(
#     pv.PolyData(points, [(3, *f) for f in trimesh.simplices]),
#     show_edges=False,
#     scalars=quad.weights,
# )
# plotter.add_mesh(normal_vecs.arrows, color="red", show_scalar_bar=False)
# plotter.show()
