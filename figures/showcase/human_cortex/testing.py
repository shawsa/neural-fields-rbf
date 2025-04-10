import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pickle
import pyvista as pv

from geodesic import GeodesicCalculator

with open("data/human_cortex_left_hemi_quad.pickle", "rb") as f:
    qf = pickle.load(f)

# x-axis is port/starbord
# y-axis is aft/fore
# z-axis is ventral/dorsal

geo = GeodesicCalculator(qf.points, qf.trimesh.simplices)

center_index = np.argmin(qf.points[:, 0])
hops = 20

plane_points, local_map = geo.to_flat(center_index, hops, verbose=True)

scatter = plt.scatter(*plane_points.T, c=geo.dist(qf.points[center_index])[local_map])
my_max = np.max(la.norm(plane_points, axis=1))
plt.xlim(-my_max, my_max)
plt.ylim(-my_max, my_max)
plt.axis("equal")


plotter = pv.Plotter(off_screen=False)
u_cmap = "viridis"
u_clim = [-1, 1]
u_mesh = pv.PolyData(
    qf.points,
    [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh["scalars"] = geo.dist(qf.points[center_index])
plotter.add_mesh(
    u_mesh,
    show_edges=True,
    lighting=True,
    scalars="scalars",
    cmap=u_cmap,
    show_scalar_bar=False,
)
plotter.add_points(
    pv.PolyData(qf.points[my_net]),
    render_points_as_spheres=True,
    scalars=np.arange(len(my_net)),
    cmap="summer",
    clim=[0, len(my_net)],
)
plotter.camera_position = (-100, 0, 0)
plotter.set_focus(qf.points[center_index])
# plotter.reset_camera()
plotter.show()
