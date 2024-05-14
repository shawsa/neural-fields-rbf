from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from scipy.spatial import distance_matrix, Delaunay

n_theta = 10
n_phi = 10
x_wavieness = 2

thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
phis = np.linspace(0, np.pi, n_phi + 2)[1:-1]

points = np.zeros((n_theta * n_phi + 2, 3))
points[-1] = (0, 0, -1)
points[-2] = (0, 0, 1)
for batch, theta in enumerate(thetas):
    rows = slice(n_phi * batch, n_phi * (batch + 1))
    points[rows, 0] = np.cos(theta + np.cos(2 * x_wavieness * phis) / n_theta) * np.sin(
        phis
    )
    points[rows, 1] = np.sin(theta) * np.sin(phis)
    points[rows, 2] = np.cos(phis)


fig = plt.figure("sphere")
ax = fig.add_subplot(projection="3d")
ax.scatter(*points.T)


