import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
import pickle
import pyvista as pv

# import pyvista as pv

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.scattered import NeuralFieldSparse
from neural_fields.firing_rate import HermiteBump

from geodesic import GeodesicCalculator

with open("data/human_cortex_left_hemi_quad.pickle", "rb") as f:
    qf = pickle.load(f)

# x-axis is port/starbord
# y-axis is aft/fore
# z-axis is ventral/dorsal

# plotter = pv.Plotter()
# plotter.add_mesh(pv.PolyData(qf.points, [(3, *t) for t in qf.trimesh.simplices]))
# plotter.add_points(pv.PolyData(qf.points[qf.points[:, 2] > 50]))
# plotter.show()


def gauss(r, sd):
    return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))


# pos_sd = 0.05
# pos_amp = 5
# neg_sd = 0.10
# neg_amp = 5

pos_sd = 3.0
pos_amp = 5
neg_sd = 6.0
neg_amp = 5


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


# firing_rate params
threshold = 0.3

gain = 20

fr_order = 4
fr_radius = threshold * 0.8


firing_rate = HermiteBump(threshold=threshold, radius=fr_radius, order=fr_order)


solver = AB5(seed=RK4(), seed_steps_per_step=2)
t0, tf = 0, 200
dt = 1e-2


geo = GeodesicCalculator(qf.points, qf.trimesh.simplices)


def my_dist(points: np.ndarray[float], point: np.ndarray[float]) -> np.ndarray[float]:
    return geo.dist(point)


# center_index = np.argmin(qf.points[:, 0])
# dists = geo.dist(qf.points[center_index])
#
# plotter = pv.Plotter()
# plotter.add_mesh(
#     pv.PolyData(qf.points, [(3, *t) for t in qf.trimesh.simplices]),
#     cmap="viridis",
#     scalars=kernel(dists),
# )
# plotter.show()


nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=firing_rate,
    weight_kernel=kernel,
    dist=my_dist,
    sparcity_tolerance=1e-5,
    verbose=True,
    tqdm_kwargs={"position": 2, "leave": False},
)

# commands for REPL use
if False:
    with open("data/human_cortex_left_hemi_nf.pickle", "wb") as f:
        pickle.dump(nf.conv_mat, f)

if False:
    with open("data/human_cortex_left_hemi_nf.pickle", "rb") as f:
        mat = pickle.load(f)


center_index = np.argmin(qf.points[:, 0])
# u0 = np.exp(-((geo.dist(qf.points[center_index]) / pos_sd) ** 2))
u0 = np.exp(-((np.abs(qf.points[:, 1]) / pos_sd) ** 2))

u = u0
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("media/cortex_labyrinth.gif")
# plotter = pv.Plotter(off_screen=False)
u_cmap = "viridis"
u_clim = [-1, 1]
u_mesh = pv.PolyData(
    qf.points,
    [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh["scalars"] = u
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
plotter.reset_camera()
# plotter.show()

time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, dt)
for index, (t, u) in enumerate(
    tqdm(
        zip(time.array, solver.solution_generator(u, nf.rhs, time)),
        total=len(time.array),
        position=0,
        leave=True,
    )
):
    if index % 100 == 0:
        u_mesh["scalars"] = u
        if index % (len(time.array) // 1000) == 0 or index == len(time.array) - 1:
            plotter.screenshot(
                f"media/cortex_labyrinth_frames/cortex_labyrinth_frames_{index}.png"
            )
        plotter.write_frame()

plotter.close()

# 3D plotting
plotter = pv.Plotter(off_screen=False)
mesh = pv.PolyData(qf.points, [(3, *f) for f in qf.trimesh.simplices])
mesh["scalars"] = u
plotter.add_mesh(
    mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="viridis",
    clim=u_clim,
    render=True,
    show_scalar_bar=False,
)
plotter.camera_position = (-100, 0, 0)
plotter.show()
