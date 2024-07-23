from itertools import product
import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree

from scipy.spatial import Voronoi, voronoi_plot_2d


class Boundary:
    def __init__(self, offset: np.ndarray[float], normal: np.ndarray[float]):
        self.offset = np.array(offset, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / la.norm(self.normal)
        self.contain_tol = 1e-15

    def __contains__(self, point: np.ndarray[float]) -> bool:
        return np.dot(self.normal, point - self.offset) >= -self.contain_tol

    def intersection(self, bnd) -> np.ndarray[float]:
        return la.solve(
            np.array([self.normal, bnd.normal]),
            np.r_[np.dot(self.normal, self.offset), np.dot(bnd.normal, bnd.offset)],
        )


class Cell:
    def __init__(
        self,
        center: np.ndarray[float],
        tree: KDTree,
        *,
        default_boundaries: tuple[Boundary] = [],
        max_neighbors: int = 20
    ):
        self.center = center
        self.tree = tree

        _, indices = tree.query(center, k=max_neighbors + 1)
        indices = indices[1:]
        neighbors = tree.data[indices]
        boundaries = [Boundary((center + nbr) / 2, center - nbr) for nbr in neighbors]
        boundaries += default_boundaries

        self.neighbor_indices = []
        for index, boundary in zip(indices, boundaries):
            verts = [
                boundary.intersection(bnd) for bnd in boundaries if bnd is not boundary
            ]
            if any(all(vert in bnd for bnd in boundaries) for vert in verts):
                self.neighbor_indices.append(index)

    @property
    def neighbors(self) -> np.ndarray[float]:
        return self.tree.data[self.neighbor_indices]

    @property
    def enum_neighbors(self):
        return zip(self.neighbor_indices, self.neighbors)


class LocalVoronoiTriangulation:
    def __init__(self, points: np.ndarray[float], borders: tuple[Boundary]):
        self.points = points
        self.borders = borders

        tree = KDTree(self.points)
        completed = set()
        self.faces = []
        for index, center in enumerate(self.points):
            cell = Cell(center, tree, default_boundaries=self.borders)
            for n1_id, n1_point in cell.enum_neighbors:
                if n1_id in completed:
                    continue
                n1 = Cell(n1_point, tree, default_boundaries=self.borders)
                for n2_id, n2_point in n1.enum_neighbors:
                    if n2_id == index or n2_id in completed:
                        continue
                    n1 = Cell(n2_point, tree, default_boundaries=self.borders)
                    if index in n1.neighbor_indices:
                        self.faces.append((index, n1_id, n2_id))
            completed.add(index)
        self.faces = np.array(self.faces, dtype=int)


np.random.seed(0)
N = 50
points = np.random.random((N, 2)) - 0.5
# center = np.r_[0.0, 0.0]
center = np.r_[-0.5, 0]
points[0] = center
tree = KDTree(points)
borders = (
    Boundary([-0.5, 0], [0.5, 0]),
    Boundary([0.5, 0], [-0.5, 0]),
    Boundary([0, -0.5], [0, 0.5]),
    Boundary([0, 0.5], [0, -0.5]),
)
cell = Cell(center, tree, default_boundaries=borders)

my_vor = LocalVoronoiTriangulation(points, borders)

vor = Voronoi(points)

import matplotlib.pyplot as plt

voronoi_plot_2d(vor)
plt.plot(*points.T, "k.")
plt.plot(*center, "k*")
plt.axis("equal")
plt.plot(*cell.neighbors.T, "gs")
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)

plt.triplot(*points.T, my_vor.faces)
