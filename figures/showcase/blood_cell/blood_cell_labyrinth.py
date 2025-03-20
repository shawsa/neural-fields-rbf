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
from neural_fields.firing_rate import Sigmoid, Heaviside, HermiteBump

from blood_cell_utils import unflatten

pointset_index = 1
N = 64_000

with open(f"data/blood_cell_{N}_quad_index{pointset_index}.pickle", "rb") as f:
    params, qf = pickle.load(f)
points = qf.points
sphere_points = unflatten(points, **params)
assert np.max(la.norm(sphere_points, axis=1)) - 1 < 1e-12


def gauss(r, sd):
    return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))


pos_sd = 0.05
pos_amp = 5
neg_sd = 0.10
neg_amp = 5


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


# firing_rate params
threshold = 0.3

gain = 20

fr_order = 4
fr_radius = threshold * 0.8


# firing_rate = Heaviside(threshold)
# firing_rate = Sigmoid(threshold, gain)
firing_rate = HermiteBump(threshold=threshold, radius=fr_radius, order=fr_order)


solver = AB5(seed=RK4(), seed_steps_per_step=2)
t0, tf = 0, 200
dt = 1e-2

nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=firing_rate,
    weight_kernel=kernel,
    sparcity_tolerance=1e-5,
    verbose=True,
    tqdm_kwargs={"position": 2, "leave": False},
)

u0 = np.zeros(len(points))
center = [0.5, 0.3]
center.append(np.sqrt(1 - center[0] ** 2 - center[1] ** 2))

center_index = np.argmin(sum((sphere_points[:, i] - center[i]) ** 2 for i in range(3)))
reflect_vec = sphere_points[center_index] - np.array([0, 0, 1])
reflect_vec /= la.norm(reflect_vec)
reflect = np.eye(3) - 2 * np.outer(reflect_vec, reflect_vec)
reflected_points = sphere_points @ reflect

thetas = np.arctan2(reflected_points[:, 1], reflected_points[:, 0])
rs = la.norm(reflected_points - np.array([0, 0, 1]), axis=1)
u0 = 5 * np.exp(-(10 * (1 * (np.cos(4 * thetas) + 3) * rs) ** 2))

u = u0

plt.ion()

top_mask = points[:, 2] > 0
modified_points = points[:, :2].copy()
modified_points[~top_mask] += np.array([2.2, 0])

figsize = (8, 4)
fig = plt.figure("flat convergence", figsize=figsize)
ax = fig.add_subplot()
scatter = ax.scatter(*modified_points.T, s=3.0, c=u, cmap="jet", vmin=-1.0, vmax=2.0)
ax.axis("off")
ax.axis("equal")

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
        scatter.set_array(u)
        fig.canvas.draw_idle()
        plt.pause(1e-5)

# 3D plotting
plotter = pv.Plotter(off_screen=True)
# plotter = pv.Plotter(off_screen=False)
shift = np.array([0, 0, 1-params["amplitude"]**2], dtype=float)
mesh = pv.PolyData(qf.points + 1.1 * shift, [(3, *f) for f in qf.trimesh.simplices])
mesh["scalars"] = u
plotter.add_mesh(
    mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="jet",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
points_inverted = qf.points.copy()
points_inverted[:, 2] *= -1
mesh2 = pv.PolyData(
    points_inverted - 1.1 * shift, [(3, *f) for f in qf.trimesh.simplices]
)
mesh2["scalars"] = u
plotter.add_mesh(
    mesh2,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="jet",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
mesh3 = pv.PolyData(qf.points + 3.2 * shift, [(3, *f) for f in qf.trimesh.simplices])
mesh["scalars"] = u
plotter.add_mesh(
    mesh3,
    show_edges=False,
    lighting=True,
    scalars=u0,
    cmap="jet",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
plotter.camera_position = "yz"
plotter.camera.elevation = 30
plotter.reset_camera()
plotter.screenshot(f"media/labyrinth_blood_cell_index{params['amplitude']}.png")
plotter.show()
