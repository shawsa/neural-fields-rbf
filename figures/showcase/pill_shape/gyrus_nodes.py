import numpy as np
import numpy.linalg as la
import pyvista as pv

from scipy.spatial import KDTree

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.sphere import SpherePoints
from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pickle


def flatten(points: np.ndarray[float]) -> np.ndarray[float]:
    points = points.copy()
    rs = la.norm(points[:, 1:], axis=1)
    points[:, 0] *= (np.tanh(2 * (rs - 1)) + 1.3) / 2.3
    # points[:, 0] *= 1 - np.exp(-rs**2)
    # points[:, 0] *= .3/(np.sqrt(1-rs**2/2))
    # points[:, 0] *= 1 - 0.5 / (1 + rs**2)
    return points


def approx_normal(
    point: np.ndarray[float],
    points: np.ndarray[float],
    tree: KDTree,
    num_neighbors=12,
):
    """Approximate the normal vector to a point using the last principle component
    of the SVD of the closest points.
    """
    stencil = tree.query(point, k=num_neighbors)[1]
    pnts = points[stencil] - point
    normal = la.svd(pnts)[2][2]
    if np.dot(point, normal) < 0:
        normal *= -1
    return normal


np.random.seed(0)
sphere = SpherePoints(64_000, auto_settle=False, verbose=True)
sphere.jostle(5_000, verbose=True)
sphere.settle(0.01, repeat=1_00, verbose=True)
vor = LocalSurfaceVoronoi(
    sphere.points,
    sphere.points,
    lambda x: la.norm(x, axis=1) - 1,
)


points = flatten(sphere.points)
tree = KDTree(points)
normals = np.empty_like(points)
for index, point in enumerate(points):
    normals[index] = approx_normal(point, points, tree, num_neighbors=50)

trimesh = TriMesh(points, vor.triangles, normals=normals)
quad = SurfaceQuad(
    trimesh=trimesh,
    rbf=PHS(3),
    poly_deg=3,
    stencil_size=32,
    verbose=True,
    tqdm_kwargs={
        "position": 0,
        "leave": True,
        "desc": "Calculating weights",
    },
)

with open("gyrus_quad.pickle", "wb") as f:
    pickle.dump(quad, f)

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
