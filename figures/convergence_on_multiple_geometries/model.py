import numpy as np

from neural_fields.firing_rate import HermiteBump


def gauss(r, sd):
    return 1 / (2 * np.pi * sd**2) * np.exp(-(r**2) / (2 * sd**2))


pos_sd = 0.05
pos_amp = 5
neg_sd = 0.10
neg_amp = 5


def kernel(r):
    return pos_amp * gauss(r, pos_sd) - neg_amp * gauss(r, neg_sd)


# firing_rate params
threshold = 0.3

gain = 20

fr_order = 10
fr_radius = threshold * 0.8


firing_rate = HermiteBump(threshold=threshold, radius=fr_radius, order=fr_order)


def initial_condition(x: np.ndarray[float]) -> np.ndarray[float]:
    return np.cos(2*x[:, 0]) * np.cos(4*x[:, 1]) / 2 + 1
