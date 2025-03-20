import numpy as np
from tqdm import tqdm
import pickle
import pyvista as pv

from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
from odeiter.adams_bashforth import AB5

from neural_fields.scattered import NeuralFieldSparse
from neural_fields.firing_rate import Sigmoid


with open("data/gyrus_quad.pickle", "rb") as f:
    qf = pickle.load(f)
points = qf.points


def implicit(x, y, z):
    rs_squared = y**2 + z**2
    rs = np.sqrt(rs_squared)
    new_x = x * 2.3 / (np.tanh(2 * (rs - 1)) + 1.3)
    return (new_x**2 + rs_squared) - 1


def gauss(r, sd):
    return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))


pos_sd = 0.05
pos_amp = 10
neg_sd = 0.1
neg_amp = 10


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


threshold = 0.2
gain = 20


solver = AB5(seed=RK4(), seed_steps_per_step=2)
t0, tf = 0, 20
dt = 1e-2

nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=Sigmoid(threshold=threshold, gain=gain),
    weight_kernel=kernel,
    sparcity_tolerance=1e-5,
    verbose=True,
    tqdm_kwargs={"position": 2, "leave": False},
)

u0 = np.zeros(len(points))
center = (0.5, 0)

top_mask = points[:, 0] > 0
thetas = np.arctan2(points[top_mask, 1] - center[0], points[top_mask, 2] - center[1])
rs = (
    (np.cos(4 * thetas) + 3)
    / 20
    * np.sqrt(
        (points[top_mask, 1] - center[0]) ** 2 + (points[top_mask, 2] - center[1]) ** 2
    )
)
u0[top_mask] = np.exp(-1600 * rs**2)

u = u0

plotter = pv.Plotter(off_screen=True)
# plotter = pv.Plotter(off_screen=False)
plotter.open_gif("media/snowflake.gif")
shift = np.array([0, 1, 0], dtype=float)
mesh = pv.PolyData(qf.points - 1.1 * shift, [(3, *f) for f in qf.trimesh.simplices])
mesh["scalars"] = u
plotter.add_mesh(
    mesh,
    show_edges=False,
    lighting=False,
    scalars="scalars",
    cmap="jet",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
points_inverted = qf.points.copy()
points_inverted[:, 0] *= -1
mesh2 = pv.PolyData(
    points_inverted + 1.1 * shift, [(3, *f) for f in qf.trimesh.simplices]
)
mesh2["scalars"] = u
plotter.add_mesh(
    mesh2,
    show_edges=False,
    lighting=False,
    scalars="scalars",
    cmap="jet",
    clim=[-2, 2],
    render=True,
    show_scalar_bar=False,
)
plotter.camera_position = "yz"
plotter.camera.elevation = -45
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
    if index % 10 == 0:
        mesh["scalars"] = u
        mesh2["scalars"] = u
        plotter.write_frame()

plotter.close()
