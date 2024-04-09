from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from numpy.fft import fft2, ifft2
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, Euler, RK4, TqdmWrapper

import imageio.v2 as imageio
import os

from neural_fields.cartesian import SpaceDomain, NeuralField
from neural_fields.kernels import Gaussian, laterally_inhibitory_weight_kernel
from neural_fields.firing_rate import Sigmoid, HermiteBump, Heaviside

from scipy.special import k0


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


class CoombesSchmidtBojakBessel:
    """The Bessel function based kernel used in
    Coombes Schmidt Bojak 2012.
    """

    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def __call__(self, r):
        r = r.copy()
        zero_mask = r <= 0
        r[zero_mask] = 1e-100
        return (
            2 / (3 * np.pi)
            * (
                k0(r)
                - k0(2 * r)
                - 1 / self.gamma * (k0(self.beta * r) - k0(2 * self.beta * r))
            )
        )


FILE_NAME = "media/snowflake7_double_res.gif"
SAVE_ANIMATION = True

# width = 60
width = 140
x_linspace_params = (-width / 2, width / 2, 801)
y_linspace_params = x_linspace_params

t0, tf = 0, 100
# t0, tf = 0, 80
delta_t = 1e-1

space = SpaceDomain(*x_linspace_params, *y_linspace_params)

# threshold = 0.1
threshold = 0.115

nf = NeuralField(
    space=space,
    # firing_rate=Heaviside(threshold=threshold),
    firing_rate=HermiteBump(threshold=threshold, radius=0.1, order=3),
    # firing_rate=Sigmoid(threshold=0.2, gain=20),
    weight_kernel=laterally_inhibitory_weight_kernel,
    # weight_kernel=CoombesSchmidtBojakBessel(beta=0.5, gamma=4)
)

time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
solver = TqdmWrapper(RK4())
# solver = TqdmWrapper(Euler())

# u0 = Gaussian(sigma=1.0).radial(np.sqrt(space.X**2 + space.Y**2) + .5*np.cos(4*np.arctan2(space.Y, space.X)))
u0 = np.zeros_like(space.X).flatten()
for index, (x, y) in enumerate(zip(space.X.ravel(), space.Y.ravel())):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if r < 5 + 2 * np.cos(7 * theta + 0*np.pi/3):
        u0[index] = 1
u0 = u0.reshape(space.X.shape)

mesh = plt.pcolormesh(space.X, space.Y, u0, cmap="jet", vmin=-2, vmax=2)
# mesh = plt.pcolormesh(space.X, space.Y, u0, cmap="jet", vmin=-.14, vmax=.4)
plt.axis("equal")
plt.colorbar()

if SAVE_ANIMATION:
    writer = imageio.get_writer(FILE_NAME, mode="I")
else:
    writer = NullContext()

with writer:
    for u in solver.solution_generator(u0, nf.rhs, time):
        mesh.set_array(u)
        plt.draw()
        plt.pause(1e-3)
        plt.savefig(FILE_NAME + ".png")
        image = imageio.imread(FILE_NAME + ".png")
        os.remove(FILE_NAME + ".png")
        writer.append_data(image)
