from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from .points import PointCloud, GaussianRepulsionKernel, ConstRepulsionKernel

# from rbf.points.points import PointCloud, GaussianRepulsionKernel, ConstRepulsionKernel
from tqdm import tqdm


def hex_limit_covering_radius(N: int):
    """
    Find the approximate covering radius for a hex grid with N points per unit area.
    """
    unit_density = 4 / (3 * np.sqrt(3))
    return np.sqrt(unit_density / N)


def hex_limit_density(h: float):
    """
    Find the approximate number of points per unit area for a hex grid with the
    covering radius h.
    """
    unit_density = 4 / (3 * np.sqrt(3))
    return ceil(1 / (unit_density * h**2))


class UnitSquare(PointCloud):
    """
    Generates N points in unit square [0, 1] x [0, 1].
    """

    def __init__(self, N: int, auto_settle=True, verbose=False):
        self.N = N
        self.h = hex_limit_covering_radius(N)
        self.n = ceil(1 / self.h)
        num_boundary = self.n * 4
        num_interior = self.N - num_boundary
        assert self.n >= 2
        points = np.empty((self.N, 2))
        boundary_points = points[num_interior:]
        side = np.linspace(0, 1, self.n, endpoint=False)
        boundary_points[: self.n][:, 0] = side
        boundary_points[: self.n][:, 1] = 0
        boundary_points[self.n : 2 * self.n][:, 0] = 1
        boundary_points[self.n : 2 * self.n][:, 1] = side
        boundary_points[2 * self.n : 3 * self.n][:, 0] = 1 - side
        boundary_points[2 * self.n : 3 * self.n][:, 1] = 1
        boundary_points[3 * self.n :][:, 0] = 0
        boundary_points[3 * self.n :][:, 1] = 1 - side

        points[:num_interior] = self.h + (1 - 2 * self.h) * np.random.random(
            (self.N - 4 * self.n, 2)
        )

        super(UnitSquare, self).__init__(
            points, num_interior=num_interior, num_boundary=num_boundary
        )

        self.const_kernel = ConstRepulsionKernel(self.h / 2)
        self.gauss_kernel = GaussianRepulsionKernel(height=1, shape=self.h / 2)
        if auto_settle:
            self.auto_settle(verbose=verbose)

    def force_shape(self, x):
        return self.h * (1 + np.tanh(-(x - self.h / 2) / self.h**2))

    def boundary_force(self, point):
        force = np.zeros_like(point)
        x, y = point
        force[0] = self.force_shape(x) - self.force_shape(1 - x)
        force[1] = self.force_shape(y) - self.force_shape(1 - y)
        return force

    def settle(self, rate: float, repeat: int = 1, verbose: bool = False):
        num_neighbors = 18
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter)
        for _ in my_iter:
            super(UnitSquare, self).settle(
                kernel=self.gauss_kernel,
                rate=rate / num_neighbors,
                num_neighbors=num_neighbors,
                force=self.boundary_force,
            )

    def jostle(self, repeat: int = 1, verbose: bool = False):
        num_neighbors = 3
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter)
        for _ in my_iter:
            super(UnitSquare, self).settle(
                kernel=self.const_kernel,
                rate=1 / num_neighbors,
                num_neighbors=num_neighbors,
                force=self.boundary_force,
            )

    def auto_settle(self, verbose=False):
        self.jostle(repeat=20, verbose=verbose)
        self.settle(rate=1, repeat=20, verbose=verbose)


if __name__ == "__main__":
    from scipy.spatial import Delaunay
    from ..geometry import circumradius, delaunay_covering_radius, triangle

    # from rbf.geometry import circumradius, delaunay_covering_radius, triangle

    plt.ion()
    N = 40_000
    unit_square = UnitSquare(N, auto_settle=False)
    # unit_square.auto_settle(verbose=True)

    (scatter,) = plt.plot(*unit_square.inner.T, "k.")
    plt.plot(*unit_square.boundary.T, "bs")

    # unit_square.jostle(repeat=20, verbose=True)
    for _ in tqdm(range(20)):
        unit_square.jostle(repeat=1)
        scatter.set_data(*unit_square.inner.T)
        plt.pause(1e-3)

    # unit_square.settle(rate=1, repeat=20, verbose=True)
    for _ in tqdm(range(20)):
        unit_square.settle(rate=1)
        scatter.set_data(*unit_square.inner.T)
        plt.pause(1e-3)

    points = unit_square.points
    mesh = Delaunay(points)
    plt.triplot(*points.T, mesh.simplices)
    circum_radii = []
    centroids = []
    for tri_indices in mesh.simplices:
        tri_points = mesh.points[tri_indices]
        centroids.append(triangle(tri_points).centroid)
        circum_radii.append(circumradius(tri_points))
    centroids = np.array(centroids)
    plt.scatter(*centroids.T, c=circum_radii)
    plt.colorbar(label="$h$")
    plt.show()

    if True:
        print("Covering Stats")
        print(f"min:\t{min(circum_radii):.3E}")
        print(
            f"median:\t{np.median(circum_radii):.3E}\t (ave={np.average(circum_radii):.3E})"
        )
        print(f"max:\t{max(circum_radii):.3E}")
        print(f"hex_limit={hex_limit_covering_radius(N):.2E}")
