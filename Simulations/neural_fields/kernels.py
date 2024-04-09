import numpy as np
import numpy.linalg as la


def laterally_inhibitory_weight_kernel(r):
    return np.exp(-r) * (2 - r)


def excitatory_weight_kernel(r):
    return np.exp(-r)


class Gaussian:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def radial(self, r):
        return (
            1
            / (2 * np.pi * self.sigma**2)
            * np.exp(-(r**2) / (2 * self.sigma**2))
        )

    def __call__(self, x):
        return self.radial(la.norm(x, axis=1))
