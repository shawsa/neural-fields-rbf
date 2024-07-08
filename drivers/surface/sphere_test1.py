import numpy as np
import numpy.linalg as la
from scipy.spatial import ConvexHull

from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

factor = 3

n_theta = factor * 20
n_phi = factor * 10
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
points /= la.norm(points, axis=1)[:, np.newaxis]

hull = ConvexHull(points)
trimesh = TriMesh(points, hull.simplices, normals=points)

rbf = PHS(3)
stencil_size = 41
poly_deg = 3

quad = SurfaceQuad(trimesh, rbf, poly_deg, stencil_size, verbose=True)
approx = np.sum(quad.weights)
exact = 4 * np.pi * 1**2
error = abs(approx - exact) / exact
print(f"{approx=}")
print(f"{exact=}")
print(f"{error=}")
