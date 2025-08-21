"""
A simple module for placing points on a sphere.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from .points import (
    PointCloud,
    GaussianRepulsionKernel,
    ConstRepulsionKernel,
)


class SpherePoints(PointCloud):
    def __init__(
        self,
        N: int,
        auto_settle: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        self.N = N
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        points = np.random.random((N, 3)) * 2 - 1
        super().__init__(
            points=points,
            num_interior=N,
            num_boundary=0,
        )
        self.project_to_sphere()

        area_per_point = 4 * np.pi / self.N
        cell_radius = np.sqrt(area_per_point / np.pi)
        self.const_kernel = ConstRepulsionKernel(cell_radius)
        self.repulsion_kernel = GaussianRepulsionKernel(
            height=1, shape=2*cell_radius
        )

        if auto_settle:
            self.auto_settle()

    def project_to_sphere(self):
        self.points /= la.norm(self.points, axis=1)[:, np.newaxis]

    @property
    def coords(self) -> tuple[np.ndarray[float]]:
        return (*self.points.T,)

    def settle(
        self,
        rate: float,
        repeat: int = 1,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        num_neighbors = 19
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter, **tqdm_kwargs)
            my_iter.set_description(f"Settling {self.N} points")
        for _ in my_iter:
            super().settle(
                kernel=self.repulsion_kernel,
                rate=rate / num_neighbors,
                num_neighbors=num_neighbors,
            )
            self.project_to_sphere()

    def jostle(self, repeat: int = 1, verbose: bool = False, tqdm_kwargs={}):
        num_neighbors = 3
        my_iter = range(repeat)
        if verbose:
            my_iter = tqdm(my_iter, **tqdm_kwargs)
            my_iter.set_description(f"Jiggling {self.N} points")
        for _ in my_iter:
            super().settle(
                kernel=self.const_kernel,
                rate=1 / num_neighbors,
                num_neighbors=num_neighbors,
            )
            self.project_to_sphere()

    def auto_settle(self):
        self.jostle(repeat=50, verbose=self.verbose, tqdm_kwargs=self.tqdm_kwargs)
        self.settle(
            rate=1, repeat=50, verbose=self.verbose, tqdm_kwargs=self.tqdm_kwargs
        )


if __name__ == "__main__":
    N = 600
    sphere = SpherePoints(N)
    plt.ion()
    ax = plt.figure().add_subplot(projection="3d")
    scatter, = ax.plot(*sphere.coords, "k.")

    for _ in tqdm(range(50)):
        sphere.jostle(repeat=1, verbose=False)
        scatter.set_data_3d(*sphere.coords)
        plt.pause(0.1)

    for _ in tqdm(range(100)):
        sphere.settle(rate=1, repeat=1, verbose=False)
        scatter.set_data_3d(*sphere.coords)
        plt.pause(0.1)
