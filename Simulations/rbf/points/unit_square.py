from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from .points import (
    PointCloud,
    GaussianRepulsionKernel,
    ConstRepulsionKernel,
)

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
        num_ghost = num_boundary + 8
        assert self.n >= 2
        points = np.empty((self.N + num_ghost, 2))

        boundary_points = points[num_interior : num_interior + num_boundary]
        side = np.linspace(0, 1, self.n, endpoint=False)
        boundary_points[: self.n][:, 0] = side
        boundary_points[: self.n][:, 1] = 0
        boundary_points[self.n : 2 * self.n][:, 0] = 1
        boundary_points[self.n : 2 * self.n][:, 1] = side
        boundary_points[2 * self.n : 3 * self.n][:, 0] = 1 - side
        boundary_points[2 * self.n : 3 * self.n][:, 1] = 1
        boundary_points[3 * self.n :][:, 0] = 0
        boundary_points[3 * self.n :][:, 1] = 1 - side

        ghost_points = points[
            num_interior + num_boundary : num_interior + num_boundary + num_ghost
        ]
        boundary_spacing = 1 / self.n
        ghost_per_side = self.n + 2
        side = np.linspace(-boundary_spacing/2, 1 + boundary_spacing/2, ghost_per_side)
        ghost_points[: ghost_per_side][:, 0] = side
        ghost_points[: ghost_per_side][:, 1] = -boundary_spacing
        ghost_points[ghost_per_side : 2 * ghost_per_side][:, 0] = 1 + boundary_spacing
        ghost_points[ghost_per_side : 2 * ghost_per_side][:, 1] = side
        ghost_points[2 * ghost_per_side : 3 * ghost_per_side][:, 0] = 1 - side
        ghost_points[2 * ghost_per_side : 3 * ghost_per_side][:, 1] = 1 + boundary_spacing
        ghost_points[3 * ghost_per_side :][:, 0] = -boundary_spacing
        ghost_points[3 * ghost_per_side :][:, 1] = 1 - side

        points[:num_interior] = self.h + (1 - 2 * self.h) * np.random.random(
            (self.N - 4 * self.n, 2)
        )

        super().__init__(
            points,
            num_interior=num_interior,
            num_boundary=num_boundary,
            num_ghost=num_ghost,
        )

        self.const_kernel = ConstRepulsionKernel(self.h / 2)
        self.repulsion_kernel = GaussianRepulsionKernel(height=1, shape=self.h)
        # self.repulsion_kernel = PowerLawRepulsionKernel(scale=1, power=2)
        if auto_settle:
            self.auto_settle(verbose=verbose)

    def force_shape(self, x):
        # return self.h * (1 + np.tanh(-(x - self.h / 2) / self.h**2))
        # return 10*self.h * (1 + np.tanh(-20 * x / self.h))
        # return self.h * np.exp(-2 / self.h * (x - self.h / 2))
        # return 0
        return (-x + self.h/2) * np.heaviside(-x, .5)

    def boundary_force(self, points):
        force = np.zeros_like(points)
        x, y = points.T
        force[:, 0] = self.force_shape(x) - self.force_shape(1 - x)
        force[:, 1] = self.force_shape(y) - self.force_shape(1 - y)
        return force

    def settle(self, rate: float, repeat: int = 1, verbose: bool = False):
        num_neighbors = 30
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter)
        for _ in my_iter:
            super().settle(
                kernel=self.repulsion_kernel,
                rate=rate / num_neighbors,
                num_neighbors=num_neighbors,
                force=self.boundary_force,
                use_ghost=True
            )

    def jostle(self, repeat: int = 1, verbose: bool = False):
        num_neighbors = 3
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter)
        for _ in my_iter:
            super().settle(
                kernel=self.const_kernel,
                rate=1 / num_neighbors,
                num_neighbors=num_neighbors,
                force=self.boundary_force,
                use_ghost=True
            )

    def auto_settle(self, verbose=False):
        self.jostle(repeat=50, verbose=verbose)
        self.settle(rate=1, repeat=50, verbose=verbose)


if __name__ == "__main__":
    from scipy.spatial import Delaunay
    from ..geometry import circumradius, delaunay_covering_radius, triangle

    if False:
        from rbf.geometry import circumradius, delaunay_covering_radius_stats, triangle
        from rbf.points.points import (
            PointCloud,
            GaussianRepulsionKernel,
            ConstRepulsionKernel,
            PowerLawRepulsionKernel,
        )

    plt.ion()
    N = 10_000
    unit_square = UnitSquare(N, auto_settle=False)
    # unit_square.auto_settle(verbose=True)

    plt.figure("Mesh")
    (scatter,) = plt.plot(*unit_square.inner.T, "k.")
    plt.plot(*unit_square.boundary.T, "bs")
    plt.plot(*unit_square.ghost.T, "or")
    plt.axis("equal")

    # unit_square.jostle(repeat=20, verbose=True)
    for _ in tqdm(range(50)):
        unit_square.jostle(repeat=1)
        scatter.set_data(*unit_square.inner.T)
        plt.pause(1e-3)

    # unit_square.settle(rate=1, repeat=20, verbose=True)
    for _ in tqdm(range(50)):
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
    plt.scatter(*centroids.T, c=circum_radii, cmap="jet")
    plt.colorbar(label="$h$")
    plt.show()

    if True:
        print("Covering Stats")
        print(f"Target: {unit_square.h:.3E}")
        print(f"min:\t{min(circum_radii):.3E}")
        print(
            f"median:\t{np.median(circum_radii):.3E}\t (ave={np.average(circum_radii):.3E})"
        )
        print(f"max:\t{max(circum_radii):.3E}")
        print(f"max/ave = {max(circum_radii)/np.average(circum_radii):.3f}")

    plt.figure("Histogram")
    plt.hist(circum_radii, bins=20)
    plt.xlabel("$h$")
