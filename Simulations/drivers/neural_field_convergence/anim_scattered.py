import matplotlib.pyplot as plt
import numpy as np
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare

from neural_fields.scattered import NeuralField, euclidian_dist, FlatTorrusDistance
from neural_fields.firing_rate import Sigmoid, HermiteBump
from neural_fields.kernels import (
    Gaussian,
    excitatory_weight_kernel,
    laterally_inhibitory_weight_kernel,
)

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


FILE_NAME = "media/scattered.gif"
SAVE_ANIMATION = False

t0, tf = 0, 50
delta_t = 1e-1
width = 30

N = 16_000
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
    firing_rate=HermiteBump(threshold=0.1, radius=0.1, order=3),
    # firing_rate=Sigmoid(threshold=0.1, gain=20),
    weight_kernel=laterally_inhibitory_weight_kernel,
    # weight_kernel=excitatory_weight_kernel,
    # dist=PeriodicDistance(x_width=width, y_width=width),
    dist=euclidian_dist,
)

time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
solver = TqdmWrapper(RK4())

u0 = (
    Gaussian(sigma=1.0)(points)
    * (2 + np.cos(np.arctan2(points[:, 1], points[:, 0])))
)

plt.figure("Solution")
scatter = plt.scatter(*points.T, c=u0, s=4, cmap="jet", vmin=-0.5, vmax=2.0)
plt.axis("equal")
plt.colorbar()


if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_NAME, mode="I")
else:
    writer = NullContext()

with writer:
    for u in solver.solution_generator(u0, nf.rhs, time):
        scatter.set_array(u)
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        os.remove(FILE_NAME + ".png")
        writer.append_data(image)
