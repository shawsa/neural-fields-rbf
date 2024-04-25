from itertools import islice
import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
from odeiter.adams_bashforth import AB5
from tqdm import tqdm

import imageio.v2 as imageio
import os


from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare

from neural_fields.scattered import (
    NeuralField,
    NeuralFieldSparse,
    FlatTorrusDistance,
)
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
SAVE_ANIMATION = False

width = 50
# t0, tf = 0, 2 * np.pi
t0, tf = 0, 0.1
# delta_t = 5e-3
delta_t = 1e-4
# delta_t = 1e-6

threshold = 0.5
gain = 5
weight_kernel_sd = 1
sol_sd = 0.5
path_radius = 10
epsilon = 0.1

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
poly_deg = 4
stencil_size = 41

points = UnitSquare(N, verbose=True).points * width - width / 2

qf = LocalQuad(
    points=points,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
)

# nf = NeuralField(
nf = NeuralFieldSparse(
    qf=qf,
    firing_rate=sol.firing,
    weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
    dist=FlatTorrusDistance(x_width=width, y_width=width),
    sparcity_tolerance=1e-16,
)


def rhs(t, u):
    return nf.rhs(t, u) + sol.rhs(*points.T, t)


time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
# solver = RK4()
solver = AB5(seed=RK4(), seed_steps_per_step=2)

u0 = sol.exact(*points.T, 0)

dot_size = 0.5

fig, (ax_sol, ax_err) = plt.subplots(1, 2, figsize=(15, 7))
scatter = ax_sol.scatter(
    *points.T,
    c=u0,
    s=dot_size,
    cmap="jet",
    vmin=sol.firing.inv(epsilon),
    vmax=np.max(u0),
)
plt.colorbar(scatter, ax=ax_sol)
err = ax_err.scatter(*points.T, c=u0 * 0 - 16, s=dot_size, cmap="jet", vmin=-10, vmax=0)
plt.colorbar(err, ax=ax_err)

for ax in (ax_sol, ax_err):
    ax.set_xlim(np.min(points[:, 0]), np.max(points[:, 0]))
    ax.set_ylim(np.min(points[:, 1]), np.max(points[:, 1]))

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_NAME, mode="I")
else:
    writer = NullContext()

frames = len(time.array) // 1
with writer:
    for t, u in islice(
        zip(
            time.array,
            tqdm_obj := tqdm(solver.solution_generator(u0, rhs, time), total=len(time.array)),
        ),
        1,
        None,
        max(1, len(time.array) // frames),
    ):
        scatter.set_array(u)
        my_exact = sol.exact(*points.T, t)
        err_arr = np.abs(u - my_exact) / my_exact
        tqdm_obj.set_description(f"error={np.max(err_arr):.3E}")
        err_arr[err_arr < 1e-16] = 1e-16
        err.set_array(np.log10(err_arr))
        plt.draw()
        plt.pause(1e-3)
        if SAVE_ANIMATION:
            plt.savefig(FILE_NAME + ".png")
            image = imageio.imread(FILE_NAME + ".png")
            os.remove(FILE_NAME + ".png")
            writer.append_data(image)
