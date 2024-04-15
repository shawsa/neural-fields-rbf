from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, Euler, RK4, TqdmWrapper

import imageio.v2 as imageio
import os

from neural_fields.cartesian import SpaceDomain, NeuralField
from neural_fields.kernels import Gaussian

from manufactured import ManufacturedSolution


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


FILE_NAME = "media/anim_cartesian_manufactured.gif"
# FILE_NAME = "media/anim_cartesian_manufactured_coarse.gif"
# FILE_NAME = "media/anim_cartesian_manufactured_euler.gif"
SAVE_ANIMATION = False

width = 50
# x_linspace_params = (-width / 2, width / 2, 61)
x_linspace_params = (-width / 2, width / 2, 401)
y_linspace_params = x_linspace_params

t0, tf = 0, 2 * np.pi
# delta_t = 1e-2
delta_t = 1e-4

space = SpaceDomain(*x_linspace_params, *y_linspace_params)

threshold = 0.1
gain = 10
weight_kernel_sd = 1
sol_sd = 0.5
path_radius = 10
epsilon = 0.01

sol = ManufacturedSolution(
    weight_kernel_sd=weight_kernel_sd,
    threshold=threshold,
    gain=gain,
    solution_sd=sol_sd,
    path_radius=path_radius,
    epsilon=epsilon,
)

print(np.max(sol.exact(space.X, space.Y, 0)))

nf = NeuralField(
    space=space,
    firing_rate=sol.firing,
    weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
)


def rhs(t, u):
    return nf.rhs(t, u) + sol.rhs(space.X, space.Y, t)


time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
# solver = TqdmWrapper(RK4())
solver = TqdmWrapper(Euler())

u0 = sol.exact(space.X, space.Y, 0)

fig, (ax_sol, ax_err) = plt.subplots(2, 1, figsize=(5, 10))
mesh = ax_sol.pcolormesh(
    space.X,
    space.Y,
    u0,
    cmap="jet",
    vmin=sol.firing.inv(epsilon),
    vmax=np.max(u0),
)
plt.colorbar(mesh, ax=ax_sol)
err = ax_err.pcolormesh(
    space.X, space.Y, space.X * 0 - 16, cmap="jet", vmin=-16, vmax=0
)
plt.colorbar(err, ax=ax_err)

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_NAME, mode="I")
else:
    writer = NullContext()

frames = 100
with writer:
    for t, u in islice(
        zip(time.array, solver.solution_generator(u0, rhs, time)),
        None,
        None,
        len(time.array) // frames,
    ):
        mesh.set_array(u)
        err_arr = np.abs(u - sol.exact(space.X, space.Y, t))
        err_arr[err_arr < 1e-16] = 1e-16
        err.set_array(np.log10(err_arr))
        plt.draw()
        plt.pause(1e-3)
        if SAVE_ANIMATION:
            plt.savefig(FILE_NAME + ".png")
            image = imageio.imread(FILE_NAME + ".png")
            os.remove(FILE_NAME + ".png")
            writer.append_data(image)
