import numpy as np
from tqdm import tqdm
import pickle
import pyvista as pv

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.scattered import NeuralFieldSparse
from neural_fields.firing_rate import Sigmoid, Heaviside


with open("data/gyrus_quad.pickle", "rb") as f:
    qf = pickle.load(f)
points = qf.points


def gauss(r, sd):
    return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))


pos_sd = 0.05
pos_amp = 10
neg_sd = 0.1
neg_amp = 15


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


# kernel_shape_factor = 20


# def kernel(r):
#     my_r = r * kernel_shape_factor
#     return kernel_shape_factor * np.exp(-my_r) * (1-my_r) * 2


# qf.weights @ kernel(la.norm(qf.points - qf.points[0], axis=1))
# rs = np.linspace(0, 2, 2001)
# plt.plot(rs, kernel(rs))

threshold = 0.2
gain = 20

# firing_rate = Sigmoid(threshold=threshold, gain=gain)
firing_rate = Heaviside(threshold)

synaptic_efficacy_timescale = 20
synaptic_depletion_rate = 2


solver = AB5(seed=RK4(), seed_steps_per_step=2)
t0, tf = 0, 2
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


v0 = np.zeros((2, len(points)))
v0[1] = 1.0
init_radius = pos_sd * 2
u_center = (0.5, 0)
q_center = (0.5, init_radius / 2)
mask = points[:, 0] > 0
v0[0][mask] = np.exp(
    -((points[mask][:, 1] - u_center[0]) ** 2 + (points[mask][:, 2] - u_center[1]) ** 2)
    / (init_radius**2)
)
v0[1][mask] = 1 - np.exp(
    -((points[mask][:, 1] - q_center[0]) ** 2 + (points[mask][:, 2] - q_center[1]) ** 2)
    / (init_radius**2)
)


v = v0.copy()

plotter = pv.Plotter(off_screen=True)
plotter.open_gif("media/synaptic_depression.gif")
# plotter = pv.Plotter(off_screen=False)
shift_vec_y = np.array([0, 1, 0], dtype=float)
shift_vec_z = np.array([0, 0, 1], dtype=float)
points_inverted = qf.points.copy()
points_inverted[:, 0] *= -1
# u plot
u_mesh = pv.PolyData(
    qf.points - 1.1 * shift_vec_y - 1.1 * shift_vec_z,
    [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh["scalars"] = v[0]
plotter.add_mesh(
    u_mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="seismic",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
# u plot mirrored
u_mesh_inverted = pv.PolyData(
    points_inverted - 1.1 * shift_vec_y + 1.1 * shift_vec_z,
    [(3, *f) for f in qf.trimesh.simplices],
)
u_mesh_inverted["scalars"] = v[0]
plotter.add_mesh(
    u_mesh_inverted,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="seismic",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
# v plot
v_mesh = pv.PolyData(
    qf.points + 1.1 * shift_vec_y - 1.1 * shift_vec_z,
    [(3, *f) for f in qf.trimesh.simplices],
)
v_mesh["scalars"] = v[1]
plotter.add_mesh(
    v_mesh,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="binary",
    clim=[0, 1],
    render=True,
    show_scalar_bar=False,
)
# v plot inverted
v_mesh_inverted = pv.PolyData(
    points_inverted + 1.1 * shift_vec_y + 1.1 * shift_vec_z,
    [(3, *f) for f in qf.trimesh.simplices],
)
v_mesh_inverted["scalars"] = v[1]
plotter.add_mesh(
    v_mesh_inverted,
    show_edges=False,
    lighting=True,
    scalars="scalars",
    cmap="binary",
    clim=[0, 1],
    render=True,
    show_scalar_bar=False,
)
plotter.camera_position = "yz"
plotter.camera.elevation = -45
plotter.reset_camera()
# plotter.show()

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
        u_mesh["scalars"] = v[0]
        v_mesh["scalars"] = v[1]
        u_mesh_inverted["scalars"] = v[0]
        v_mesh_inverted["scalars"] = v[1]
        # plotter.update_scalars(v[0], render=False)
        plotter.write_frame()

plotter.close()
