from itertools import combinations
import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree


class Boundary:
    def __init__(self, offset: np.ndarray[float], normal: np.ndarray[float]):
        self.offset = np.array(offset, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / la.norm(self.normal)
        self.contain_tol = 1e-15

    def __contains__(self, point: np.ndarray[float]) -> bool:
        return np.dot(self.normal, point - self.offset) >= -self.contain_tol


def parallel(bnd1: Boundary, bnd2: Boundary, tol=1e-14) -> bool:
    return abs(np.dot(bnd1.normal, bnd2.normal)) > 1 - tol


def intersection2D(bnd1: Boundary, bnd2: Boundary) -> None | np.ndarray[float]:
    try:
        return la.solve(
            np.array([bnd1.normal, bnd2.normal]),
            np.r_[np.dot(bnd1.normal, bnd1.offset), np.dot(bnd2.normal, bnd2.offset)],
        )
    except la.LinAlgError:
        return None


def intersections2D(*boundaries: tuple[Boundary]):
    """Find all intersections of pairs of boundaries."""
    n = len(boundaries)
    assert n >= 2
    mat = np.array(
        [[bnd1.normal, bnd2.normal] for bnd1, bnd2 in combinations(boundaries, 2)]
    )
    rhs = np.array(
        [
            [np.dot(bnd1.normal, bnd1.offset), np.dot(bnd2.normal, bnd2.offset)]
            for bnd1, bnd2 in combinations(boundaries, 2)
        ]
    )
    mask = ~np.array(
        [parallel(bnd1, bnd2) for bnd1, bnd2 in combinations(boundaries, 2)]
    )
    return la.solve(mat[mask], rhs[mask])


def non_paraellel_pairs(normals: np.ndarray[float], tol=1e-14) -> np.ndarray[bool]:
    mask = np.abs(normals @ normals.T) < 1 - tol
    return mask


def contained_mat(points: np.ndarray[float], boundaries: tuple[Boundary], tol=1e-14):
    normals = np.array([bnd.normal for bnd in boundaries])
    offsets = np.array([bnd.offset for bnd in boundaries])
    return (
        sum(
            normals[:, i] * np.subtract.outer(points[:, i], offsets[:, i])
            for i in range(normals.shape[-1])
        )
        > -tol
    )


class Cell:
    def __init__(
        self,
        center: np.ndarray[float],
        tree: KDTree,
        *,
        default_boundaries: tuple[Boundary] = [],
        max_neighbors: int = 20,
    ):
        self.center = center
        self.tree = tree

        self.find_neighbors(max_neighbors, default_boundaries)

    def find_neighbors(self, max_neighbors, default_boundaries):
        _, indices = tree.query(center, k=max_neighbors + 1)
        neighbors = tree.data[indices[1:]]
        boundaries = [Boundary((center + nbr) / 2, center - nbr) for nbr in neighbors]
        boundaries += default_boundaries
        pnts = intersections2D(*boundaries)
        # self.verts =

    # def find_neighbors(self, max_neighbors, default_boundaries):
    #     _, indices = tree.query(center, k=max_neighbors + 1)
    #     indices = indices[1:]
    #     neighbors = tree.data[indices]
    #     boundaries = [Boundary((center + nbr) / 2, center - nbr) for nbr in neighbors]
    #     boundaries += default_boundaries

    #     self.neighbor_indices = []
    #     for index, boundary in zip(indices, boundaries):
    #         verts = [
    #             intersection2D(boundary, bnd)
    #             for bnd in boundaries
    #             if bnd is not boundary
    #         ]
    #         verts = [vert for vert in verts if vert is not None]
    #         if any(all(vert in bnd for bnd in boundaries) for vert in verts):
    #             self.neighbor_indices.append(index)

    @property
    def neighbors(self) -> np.ndarray[float]:
        return self.tree.data[self.neighbor_indices]


class LocalVoronoiTriangulation:
    def __init__(self, points: np.ndarray[float], borders: tuple[Boundary]):
        self.points = points
        self.borders = borders

        self.tree = KDTree(self.points)
        self.init_cells()
        self.init_triangles()

    def init_cells(self):
        self.cells = [Cell(point, self.tree) for point in self.points]

    def init_triangles(self):
        dont_check = set()
        triangles = []
        for index, cell in enumerate(self.cells):
            dont_check.add(index)
            for n1 in cell.neighbor_indices:
                if n1 in dont_check:
                    continue
                for n2 in self.cells[n1].neighbor_indices:
                    if n2 in dont_check:
                        continue
                    if index in self.cells[n2].neighbor_indices:
                        triangles.append((index, n1, n2))
        self.triangles = np.array(triangles, dtype=int)


# class Planar(Cell):


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .unit_square import UnitSquare

    # np.random.seed(0)
    N = 200
    points = UnitSquare(N).points
    tree = KDTree(points)
    _, (_, center_index) = tree.query(np.r_[0.5, 0.5], k=2)
    center = points[center_index]
    borders = (
        Boundary([1, 0], [-1, 0]),
        Boundary([0, 1], [0, -1]),
        Boundary([0, 0], [1, 0]),
        Boundary([0, 0], [0, 1]),
    )
    cell = Cell(center, tree, default_boundaries=borders)

    my_vor = LocalVoronoiTriangulation(points, borders)

    # vor = Voronoi(points)
    # voronoi_plot_2d(vor)

    plt.plot(*points.T, "k.")
    plt.plot(*center, "k*")
    plt.axis("equal")
    plt.plot(*cell.neighbors.T, "gs")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)

    plt.triplot(*points.T, my_vor.triangles)

    test_boundaries = [*borders, Boundary([0.5, 0.5], [-1, 0])]
    pnts = intersections2D(*test_boundaries)
    plt.plot(*pnts.T, "k.")

    contained_mat(pnts, test_boundaries)
    plt.plot(*pnts[inside].T, "g*")
