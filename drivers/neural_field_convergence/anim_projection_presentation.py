import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm

from manufactured import ManufacturedSolution
from neural_fields.cartesian import SpaceDomain
from odeiter.time_domain import TimeDomain_Start_Stop_Steps
from rbf.interpolate import LocalInterpolator
from rbf.points import UnitSquare
from rbf import PHS

import imageio.v2 as imageio
import os


class NullContext:
    """
    A context that does nothing. Used an an adapter pattern to turn
    off saving the animation.
    """

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def append_data(self, *args):
        pass


FILE_FULL = "media/presentation_projection_full.gif"
FILE_POINT = "media/presentation_projection_points.gif"
FILE_INTERP = "media/presentation_projection_interp.gif"
FILE_CONV_0 = "media/presentation_projection_convergence_0.gif"
FILE_CONV_1 = "media/presentation_projection_convergence_1.gif"
FILE_CONV_2 = "media/presentation_projection_convergence_2.gif"


TEMP_FILE = "media/temp"
SAVE_ANIMATION = False

figsize = (4, 4)

num_points_per_side = 401
num_steps = 100

width = 30
x_linspace_params = (-width / 2, width / 2, num_points_per_side)
y_linspace_params = x_linspace_params
space = SpaceDomain(*x_linspace_params, *y_linspace_params)

t0, tf = 0, 2 * np.pi
time = TimeDomain_Start_Stop_Steps(t0, tf, num_steps)

threshold = 0.5
gain = 5
weight_kernel_sd = 1
sol_sd = 2.0
path_radius = 5
epsilon = 0.1
sol = ManufacturedSolution(
    weight_kernel_sd=weight_kernel_sd,
    threshold=threshold,
    gain=gain,
    solution_sd=sol_sd,
    path_radius=path_radius,
    epsilon=epsilon,
)

u0 = sol.exact(space.X, space.Y, t0)

# Exact
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("equal")
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_FULL, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in time.array:
        mesh.set_array(sol.exact(space.X, space.Y, t))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)

# Points
num_points = 100
np.random.seed(0)
points = UnitSquare(num_points, verbose=True).points * width - width / 2

plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    alpha=0.5,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
scat = plt.scatter(
    *points.T,
    c=sol.exact(*points.T, t0),
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_POINT, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in time.array:
        mesh.set_array(sol.exact(space.X, space.Y, t))
        scat.set_array(sol.exact(*points.T, t))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)

# Projected
rbf = PHS(7)
poly_deg = 4
stencil_size = 35

plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.plot(*points.T, "g.")
vor = Voronoi(points, furthest_site=False)
voronoi_plot_2d(
    vor,
    ax=plt.gca(),
    show_points=False,
    show_vertices=False,
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_INTERP, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in (tqdm_obj := tqdm(time.array, position=0, leave=True)):
        approx = LocalInterpolator(
            points=points,
            fs=sol.exact(*points.T, t),
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
        )
        mesh.set_array(approx(space.points))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)

# convergence 0
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_CONV_0, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in (tqdm_obj := tqdm(time.array, position=0, leave=True)):
        approx = LocalInterpolator(
            points=points,
            fs=sol.exact(*points.T, t),
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
        )
        mesh.set_array(approx(space.points))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)

# convergence 1
num_points = 150
np.random.seed(0)
points = UnitSquare(num_points, verbose=True).points * width - width / 2
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_CONV_1, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in (tqdm_obj := tqdm(time.array, position=0, leave=True)):
        approx = LocalInterpolator(
            points=points,
            fs=sol.exact(*points.T, t),
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
        )
        mesh.set_array(approx(space.points))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)

# convergence 2
num_points = 200
np.random.seed(0)
points = UnitSquare(num_points, verbose=True).points * width - width / 2
plt.figure(figsize=figsize)
mesh = plt.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=np.min(u0),
    vmax=np.max(u0),
)
plt.xlim(-width / 2, width / 2)
plt.ylim(-width / 2, width / 2)
plt.axis("off")

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_CONV_2, mode="I", loop=0)
else:
    writer = NullContext()

with writer:
    for t in (tqdm_obj := tqdm(time.array, position=0, leave=True)):
        approx = LocalInterpolator(
            points=points,
            fs=sol.exact(*points.T, t),
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
        )
        mesh.set_array(approx(space.points))
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(TEMP_FILE + ".png")
        image = imageio.imread(TEMP_FILE + ".png")
        os.remove(TEMP_FILE + ".png")
        writer.append_data(image)
