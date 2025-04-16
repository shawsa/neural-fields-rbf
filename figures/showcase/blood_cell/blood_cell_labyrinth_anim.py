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

from blood_cell_utils import unflatten

pointset_index = 2
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


geo = GeodesicCalculator(qf.points)


def arc_from_chord(
    points: np.ndarray[float], point: np.ndarray[float]
) -> np.ndarray[float]:
    angles = np.arccos(points @ point)
    return angles


if pointset_index == 0:
    my_dist = arc_from_chord
    print("using arclength distance")
else:

    def my_dist(
        points: np.ndarray[float], point: np.ndarray[float]
    ) -> np.ndarray[float]:
        return geo.dist(point)


nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=firing_rate,
    weight_kernel=kernel,
    dist=my_dist,
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


plotter = pv.Plotter(off_screen=True)
plotter.open_gif(f"media/labyrinth_{pointset_index}.gif")
# plotter = pv.Plotter(off_screen=False)
shift_points = qf.points.copy()
shift_points[:, 2] *= -1
shift_points[:, 0] += 2.1
cmap = "viridis"
clim = [-1, 1]
u_mesh = pv.PolyData(
    qf.points, [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh["scalars"] = u
plotter.add_mesh(
    u_mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap=cmap,
    clim=clim,
    render=True,
    show_scalar_bar=False,
)
shift_mesh = pv.PolyData(
    shift_points,
    [(3, *f) for f in qf.trimesh.simplices],
)
shift_mesh["scalars"] = u
plotter.add_mesh(
    shift_mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap=cmap,
    clim=clim,
    render=True,
    show_scalar_bar=False,
)
plotter.camera_position = "xy"
plotter.camera.elevation = -55.0
plotter.reset_camera()
# plotter.show()

plt.ion()

top_mask = points[:, 2] > 0
modified_points = points[:, :2].copy()
modified_points[~top_mask] += np.array([2.6, 0])

figsize = (8, 8)
fig = plt.figure("flat convergence", figsize=figsize)
ax = fig.add_subplot()
scatter_u = ax.scatter(
    *modified_points.T, s=3.0, c=u, cmap="jet", vmin=-1.0, vmax=2.0
)
ax.axis("off")
ax.axis("equal")
plt.pause(1e-5)

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
        scatter_u.set_array(u)
        fig.canvas.draw_idle()
        plt.pause(1e-5)
        u_mesh["scalars"] = u
        shift_mesh["scalars"] = u
        plotter.write_frame()

plotter.close()
