import meshlib.mrmeshpy as mesh
import meshlib.mrmeshnumpy as meshn
import numpy as np
import numpy.linalg as la
import pickle
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi

from geodesic import GeodesicCalculator


with open("data/bumpy_sphere_64000_quad.pickle", "rb") as f:
    params, qf = pickle.load(f)

    sphere_points = qf.points / la.norm(qf.points, axis=1)[:, np.newaxis]

vor = LocalSurfaceVoronoi(
    sphere_points,
    sphere_points,
    lambda x: la.norm(x, axis=1) - 1,
)

my_points = qf.points * la.norm(qf.points, axis=1)[:, np.newaxis]**5

geo = GeodesicCalculator(my_points)

plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(my_points, [(3, *f) for f in vor.triangles]),
    show_edges=False,
    scalars=geo.dist(my_points[0]),
    cmap="jet",
)
plotter.add_points(my_points[0], color="white", render_points_as_spheres=True)
plotter.show()
