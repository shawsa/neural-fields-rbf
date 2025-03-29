import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm
import pickle
import pyvista as pv
from pyvista.core.utilities import lines_from_points

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.scattered import NeuralFieldSparse
from neural_fields.firing_rate import HermiteBump

from bumpy_sphere_utils import rotation_matrix

from min_energy_points.sphere import SpherePoints

from geodesic import GeodesicCalculator

SAVE_3D_ANIMATION = True

N = 64_000

with open(f"data/bumpy_sphere_{N}_quad.pickle", "rb") as f:
    params, qf = pickle.load(f)
points = qf.points


def gauss(r, sd, normalized=True):
    if normalized:
        return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))
    else:
        return np.exp(-(r**2) / (2 * sd**2))


pos_sd = 0.05
pos_amp = 5
neg_sd = 0.1
neg_amp = 7


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


threshold = 0.3
fr_order = 4
fr_radius = threshold * 0.8
firing_rate = HermiteBump(threshold=threshold, radius=fr_radius, order=fr_order)

synaptic_efficacy_timescale = 20
synaptic_depletion_rate = 2

solver = AB5(seed=RK4(), seed_steps_per_step=2)
t0, tf = 0, 1600
dt = 1e-2

geo = GeodesicCalculator(qf.points)


def my_dist(points: np.ndarray[float], point: np.ndarray[float]) -> np.ndarray[float]:
    return geo.dist(point)


nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=firing_rate,
    weight_kernel=kernel,
    dist=my_dist,
    sparcity_tolerance=1e-5,
    verbose=True,
    tqdm_kwargs={"position": 0, "leave": True},
)


def rhs(t, vec):
    u, q = vec
    fu = firing_rate(u)
    v = np.empty_like(vec)
    v[0] = -u + nf.conv_mat @ (fu * q)
    v[1] = (1 - q - synaptic_depletion_rate * q * fu) / synaptic_efficacy_timescale
    return v


sphere_points = points.copy()
sphere_points /= la.norm(sphere_points, axis=1)[:, np.newaxis]


num_bumps = 25
q_center_dist = pos_sd
np.random.seed(0)
v0 = np.zeros((2, len(points)))
# v0[1] = 1.0
u_centers = SpherePoints(num_bumps).points
q_angles = np.random.random(num_bumps) * 2 * np.pi
q_centers = [
    np.array([q_center_dist * np.cos(angle), q_center_dist * np.sin(angle), 1.0])
    @ rotation_matrix(u_center)
    for u_center, angle in zip(u_centers, q_angles)
]
q_centers = [qc / la.norm(qc) for qc in q_centers]
v0[0] = sum(
    gauss(la.norm(sphere_points - u_center, axis=1), pos_sd, normalized=False)
    for u_center in u_centers
)
v0[1] = 1 - sum(
    gauss(la.norm(sphere_points - q_center, axis=1), pos_sd, normalized=False)
    for q_center in q_centers
)

v = v0.copy()

if SAVE_3D_ANIMATION:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif("media/many_spots.gif")
    # plotter = pv.Plotter(off_screen=False)
    shift_vec_x = np.array([1, 0, 0], dtype=float)
    shift_vec_y = np.array([0, 1, 0], dtype=float)
    shift_amount = 1.3
    u_cmap = "viridis"
    u_clim = [-1, 1]
    v_cmap = "binary"
    v_clim = [0, 1.5]
    u_mesh = pv.PolyData(
        qf.points - shift_amount * shift_vec_x,
        [(3, *f) for f in qf.trimesh.simplices],
    )
    u_mesh["scalars"] = v[0]
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
    v_mesh = pv.PolyData(
        qf.points + shift_amount * shift_vec_x,
        [(3, *f) for f in qf.trimesh.simplices],
    )
    v_mesh["scalars"] = v[1]
    plotter.add_mesh(
        v_mesh,
        show_edges=False,
        lighting=True,
        scalars="scalars",
        cmap=v_cmap,
        clim=v_clim,
        render=True,
        show_scalar_bar=False,
    )
    plotter.camera_position = "xy"
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
    *modified_points.T, s=3.0, c=v[0], cmap="jet", vmin=-1.0, vmax=2.0
)
scatter_v = ax.scatter(
    *(modified_points + np.array([0, 2.6])).T,
    s=3.0,
    c=v[1],
    cmap="jet",
    vmin=-1.0,
    vmax=2.0,
)
ax.axis("off")
ax.axis("equal")
plt.pause(1e-5)

time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, dt)
for index, (t, v) in enumerate(
    tqdm(
        zip(time.array, solver.solution_generator(v, rhs, time)),
        total=len(time.array),
        position=0,
        leave=True,
    )
):
    if index % 100 == 0:
        scatter_u.set_array(v[0])
        scatter_v.set_array(v[1])
        fig.canvas.draw_idle()
        plt.pause(1e-5)
        if SAVE_3D_ANIMATION:
            u_mesh["scalars"] = v[0]
            v_mesh["scalars"] = v[1]
            if index % (len(time.array) // 100) == 0 or index == len(time.array) - 1:
                plotter.screenshot(
                    f"media/many_spot_frames/many_spot_anim{index}.png"
                )
            plotter.write_frame()

if SAVE_3D_ANIMATION:
    plotter.close()
