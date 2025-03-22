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
t0, tf = 0, 800
dt = 1e-2

nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=firing_rate,
    weight_kernel=kernel,
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


def rotation_matrix(point: np.ndarray[float]) -> np.ndarray[float]:
    """
    Get a matrix that rotates the given point to the z-axis.
    """
    x, y, z = point / la.norm(point)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2)) - np.pi / 2
    R1 = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    R2 = np.array(
        [
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ]
    )
    return R2 @ R1


v0 = np.zeros((2, len(points)))
v0[1] = 1.0
u_center = [0.5, 0.0]
q_center = [u_center[0], u_center[1] + pos_sd]
u_center.append(np.sqrt(1 - u_center[0] ** 2 - u_center[1] ** 2))
q_center.append(np.sqrt(1 - q_center[0] ** 2 - q_center[1] ** 2))
v0[0] = gauss(la.norm(sphere_points - u_center, axis=1), pos_sd, normalized=False)
v0[1] = 1 - gauss(la.norm(sphere_points - q_center, axis=1), pos_sd, normalized=False)

v = v0.copy()

if SAVE_3D_ANIMATION:
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif("media/synaptic_depression.gif")
    # plotter = pv.Plotter(off_screen=False)
    shift_vec_x = np.array([1, 0, 0], dtype=float)
    shift_vec_y = np.array([0, 1, 0], dtype=float)
    shift_amount = 1.3
    u_cmap = "jet"
    u_clim = [-1, 1]
    v_cmap = "binary"
    v_clim = [0, 1.5]
    points_inverted = qf.points.copy()
    points_inverted[:, 2] *= -1
    max_index = np.argmax(v[0])
    my_rot = rotation_matrix(points[max_index])
    my_points = qf.points @ my_rot.T
    # u plot
    u_mesh = pv.PolyData(
        my_points - shift_amount * shift_vec_x,
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
        my_points + shift_amount * shift_vec_x,
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
    trail_mesh = lines_from_points(
        points[max_index] @ my_rot.T + shift_amount * shift_vec_x
    )
    plotter.add_mesh(
        trail_mesh,
        color="red",
        name="trail",
        line_width=5,
    )
    # plotter.camera_position = "yz"
    # plotter.camera.elevation = 70
    # plotter.reset_camera()
    plotter.camera_position = "xy"
    # plotter.camera.elevation = 0
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
peak_indices = []
smooth_peaks = []


blur_filter = np.exp(-np.linspace(0, 2, 1001))[::-1]
blur_filter /= sum(blur_filter)


def blur(points):
    if len(points) < len(blur_filter):
        ave = sum(points * blur_filter[-len(points) :, np.newaxis]) / sum(
            blur_filter[-len(points) :]
        )
    else:
        ave = sum(points[-len(blur_filter) :] * blur_filter[:, np.newaxis])
    return ave


# smooth_peaks = [
#     blur(qf.points[peak_indices][:i]) for i in range(1, len(peak_indices) + 1)
# ]


for index, (t, v) in enumerate(
    tqdm(
        zip(time.array, solver.solution_generator(v, rhs, time)),
        total=len(time.array),
        position=0,
        leave=True,
    )
):
    peak_indices.append(np.argmax(v[0]))
    smooth_peaks.append(blur(qf.points[peak_indices]))
    if index % 100 == 0:
        scatter_u.set_array(v[0])
        scatter_v.set_array(v[1])
        fig.canvas.draw_idle()
        plt.pause(1e-5)
        if SAVE_3D_ANIMATION:
            u_mesh["scalars"] = v[0]
            v_mesh["scalars"] = v[1]
            # u_mesh_inverted["scalars"] = v[0]
            # v_mesh_inverted["scalars"] = v[1]
            rot_mat = rotation_matrix(smooth_peaks[-1])
            my_points = qf.points @ rot_mat.T
            u_mesh.points = my_points - shift_amount * shift_vec_x
            v_mesh.points = my_points + shift_amount * shift_vec_x
            # plotter = pv.Plotter()
            smooth_arr = np.array(smooth_peaks)
            smooth_arr += 0.05 * smooth_arr / la.norm(smooth_arr, axis=1)[:, np.newaxis]
            trail_mesh = lines_from_points(
                smooth_arr @ rot_mat.T + shift_amount * shift_vec_x
            )
            plotter.add_mesh(
                trail_mesh,
                color="red",
                name="trail",
                line_width=5,
            )
            # plotter.show()
            plotter.write_frame()

if SAVE_3D_ANIMATION:
    plotter.close()

with open("data/bump_path.pickle", "wb") as f:
    pickle.dump(peak_indices, f)
