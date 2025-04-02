import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree

from min_energy_points.sphere import SpherePoints
from min_energy_points import LocalSurfaceVoronoi

from rbf.rbf import PHS
from rbf.surface import approximate_normals

import pyvista as pv

N = 50_000
np.random.seed(0)
sphere_points = SpherePoints(N, auto_settle=False)
sphere_points.jostle(5_000, verbose=True)
sphere_points.settle(0.01, repeat=100, verbose=True)
points = sphere_points.points


k = 35
rbf = PHS(9)
poly_deg = 5
normals = approximate_normals(points, rbf, poly_deg, stencil_size=k)
normals *= np.sign(np.dot(normals[0], points[0]))

print(np.max(la.norm(points - normals, axis=1)))
print(np.max(np.arccos(np.sum(points * normals, axis=1))))

vor = LocalSurfaceVoronoi(
    points,
    points,
    lambda x: la.norm(x, axis=1) - 1,
)

plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(points, [(3, *f) for f in vor.triangles]),
)
plotter.show()
