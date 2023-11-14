import matplotlib.pyplot as plt
import numpy as np
from .points import PointCloud, GaussianRepulsionKernel
from tqdm import tqdm


# make a square with N x N points


class UnitSquare(PointCloud):
    """
    Generates N^2 points in unit square [0, 1] x [0, 1] with N points along each
    boundary.
    """

    def __init__(self, N: int, auto_settle=True, verbose=False):
        self.N = N
        xs, ys = np.meshgrid(*2 * (np.linspace(0, 1, N),))
        points = np.block([[xs.ravel()], [ys.ravel()]]).T
        boundary_mask = (
            (points[:, 0] == 0.0)
            | (points[:, 0] == 1.0)
            | (points[:, 1] == 0.0)
            | (points[:, 1] == 1.0)
        )

        num_boundary = (N - 1) * 4
        sorted_points = np.empty_like(points)
        sorted_points[:-num_boundary] = points[~boundary_mask]
        sorted_points[-num_boundary:] = points[boundary_mask]
        super(UnitSquare, self).__init__(
            sorted_points, num_interior=N**2 - num_boundary, num_boundary=num_boundary
        )
        if auto_settle:
            self.auto_settle(verbose=verbose)

    def force_shape(self, x):
        return (1 + np.tanh(-self.N*2 * x)) / 2

    def boundary_force(self, point):
        force = np.zeros_like(point)
        x, y = point
        force[0] = self.force_shape(x) - self.force_shape(1 - x)
        force[1] = self.force_shape(y) - self.force_shape(1 - y)
        return 4 * force

    def settle(self, rate: float, repeat: int = 1, verbose=False):
        kernel = GaussianRepulsionKernel(height=4, shape=2 / self.N)
        # rate = 0.2 / N
        # radius = 3.1 / N
        num_neighbors = 18
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter)
        for _ in my_iter:
            super(UnitSquare, self).settle(
                kernel=kernel,
                rate=rate / self.N,
                num_neighbors=num_neighbors,
                force=self.boundary_force,
            )

    def auto_settle(self, verbose=False):
        for rate, repeat in [
            (2, 100),
            (0.1, 100),
            (0.05, 100),
        ]:
            self.settle(rate=rate, repeat=repeat, verbose=verbose)


if __name__ == "__main__":
    plt.ion()
    N = 45
    points = UnitSquare(N)
    points.auto_settle()

    (scatter,) = plt.plot(*points.inner.T, "k.")
    plt.plot(*points.boundary.T, "bs")

    for rate, repeat in [
        (2, 100),
        (0.1, 100),
        (0.05, 100),
    ]:
        for _ in tqdm(range(repeat)):
            points.settle(rate=rate)
            scatter.set_data(*points.inner.T)
            plt.pause(1e-3)
