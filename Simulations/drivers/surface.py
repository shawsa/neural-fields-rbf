from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix, Delaunay, ConvexHull


class TriMesh:
    def __init__(self, points: np.ndarray[float], simplices: np.ndarray[int]):
        self.points = points
        self.simplices = simplices

    def neighbors(self, face_index: int):
        a, b, c = self.simplices[face_index]
        neighbors = []
        for face in self.simplices:
            if sum([a in face, b in face, c in face]) == 2:
                neighbors.append(face)
        return np.array(neighbors, dtype=int)


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


hull = ConvexHull(points)
trimesh = TriMesh(points, hull.simplices)

fig = plt.figure("sphere")
ax = fig.add_subplot(projection="3d")
ax.scatter(*points.T)
# ax.add_collection(Poly3DCollection(points[hull.simplices]))

for face in hull.simplices:
    indices = np.array((*face, face[0]), dtype=int)
    ax.plot(*points[indices].T, "k-")


face_index = 45
face = trimesh.simplices[face_index]
ax.add_collection(Poly3DCollection(points[face][None, :], facecolors="blue"))
ax.add_collection(
    Poly3DCollection(points[trimesh.neighbors(face_index)], facecolors="green")
)
