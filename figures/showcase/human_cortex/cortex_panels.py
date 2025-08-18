import pickle
import pyvista as pv

with open("data/human_cortex_left_hemi_quad.pickle", "rb") as f:
    qf = pickle.load(f)
    points = qf.points

with open("data/cortex_stripe_solution.pickle", "rb") as f:
    sol = pickle.load(f)

# x-axis is port/starbord
# y-axis is aft/fore
# z-axis is ventral/dorsal

plotter = pv.Plotter(off_screen=False)
u_cmap = "viridis"
u_clim = [-1, 1]
u_mesh = pv.PolyData(
    qf.points,
    [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh["scalars"] = qf.weights
plotter.add_mesh(
    u_mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap=u_cmap,
    clim=u_clim,
    render=True,
    show_scalar_bar=False,
)
# plotter.camera_position = "yz"
plotter.camera_position = (-100, 0, 0)
plotter.show()
