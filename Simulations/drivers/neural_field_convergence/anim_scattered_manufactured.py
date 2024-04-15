from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, Euler, RK4, TqdmWrapper

import imageio.v2 as imageio
import os


from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare

from neural_fields.scattered import NeuralField, euclidian_dist, FlatTorrusDistance
from neural_fields.firing_rate import Sigmoid, HermiteBump
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


FILE_NAME = "media/anim_scattered_manufactured.gif"
SAVE_ANIMATION = True

width = 50
t0, tf = 0, 2 * np.pi
delta_t = 1e-2
# delta_t = 1e-4

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

N = 64_000
# N = 16_000
rbf = PHS(3)
poly_deg = 2
stencil_size = 21

points = UnitSquare(N, verbose=True).points * width - width / 2

qf = LocalQuad(
    points=points,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
)

nf = NeuralField(
    qf=qf,
    firing_rate=sol.firing,
    weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
    dist=FlatTorrusDistance(x_width=width, y_width=width),
)


def rhs(t, u):
    return nf.rhs(t, u) + sol.rhs(*points.T, t)


time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
solver = TqdmWrapper(RK4())
# solver = TqdmWrapper(Euler())

u0 = sol.exact(*points.T, 0)

dot_size = 0.5

fig, (ax_sol, ax_err) = plt.subplots(2, 1, figsize=(5, 10))
scatter = ax_sol.scatter(
    *points.T,
    c=u0,
    s=dot_size,
    cmap="jet",
    vmin=sol.firing.inv(epsilon),
    vmax=np.max(u0),
)
plt.colorbar(scatter, ax=ax_sol)
err = ax_err.scatter(*points.T, c=u0 * 0 - 16, s=dot_size, cmap="jet", vmin=-16, vmax=0)
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
        scatter.set_array(u)
        err_arr = np.abs(u - sol.exact(*points.T, t))
        err_arr[err_arr < 1e-16] = 1e-16
        err.set_array(np.log10(err_arr))
        plt.draw()
        plt.pause(1e-3)
        if SAVE_ANIMATION:
            plt.savefig(FILE_NAME + ".png")
            image = imageio.imread(FILE_NAME + ".png")
            os.remove(FILE_NAME + ".png")
            writer.append_data(image)
