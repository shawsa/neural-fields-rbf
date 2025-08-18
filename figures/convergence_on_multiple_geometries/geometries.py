from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from min_energy_points.sphere import SpherePoints
from min_energy_points.torus import SpiralTorus
from min_energy_points.points import PointCloud
from min_energy_points.kernels import (
    ConstRepulsionKernel,
    GaussianRepulsionKernel,
)
from min_energy_points.local_voronoi import LocalSurfaceVoronoi


class Shape(ABC):
    @abstractproperty
    def points(self):
        raise NotImplementedError

    @abstractproperty
    def normals(self):
        raise NotImplementedError

    @abstractmethod
    def implicit_surf(self):
        """A function from R^3 -> R for which points on the surface get
        mapped to 0.
        """
        raise NotImplementedError


class Sphere(Shape):
    def __init__(self, N: int, verbose: bool = False, tqdm_kwargs={}):
        pass
        self.sphere = SpherePoints(
            N, auto_settle=False, verbose=verbose, tqdm_kwargs=tqdm_kwargs
        )
        self.sphere.jostle(5_00, verbose=verbose)
        self.sphere.settle(0.005, repeat=1_00, verbose=verbose)

    @property
    def points(self):
        return self.sphere.points

    @property
    def normals(self):
        return self.sphere.points

    def implicit_surf(self, x: np.ndarray[float]):
        """A function from R^3 -> R for which points on the surface get
        mapped to 0.
        """
        return la.norm(x, axis=1) - 1


class DupinCyclide(PointCloud, Shape):
    @staticmethod
    def parameteric(us, vs, *, a, b, c, d):
        assert len(us) == len(vs)
        denom = a - c * np.cos(us) * np.cos(vs)
        points = np.zeros((len(us), 3))
        points[:, 0] = (
            d * (c - a * np.cos(us) * np.cos(vs)) + b**2 * np.cos(us)
        ) / denom
        points[:, 1] = b * np.sin(us) * (a - d * np.cos(vs)) / denom
        points[:, 2] = b * np.sin(vs) * (c * np.cos(us) - d) / denom
        return points

    def __init__(
        self,
        N: int,
        *,
        # a: float = 1,
        # b: float = 0.98,
        # c: float = 0.1983,
        # d: float = 0.5,
        auto_settle: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        # hard coding parameter values
        a = 1
        b = 0.98
        c = 0.1983
        d = 0.5
        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        points = np.zeros((N, 3))
        us = np.linspace(-np.pi, np.pi, N, endpoint=False)
        vs = 2 * np.pi * np.random.random(N)
        points = DupinCyclide.parameteric(us, vs, a=a, b=b, c=c, d=d)
        points = self.projection(points)
        super().__init__(
            all_points=points,
            num_fixed=0,
            num_ghost=0,
        )

        # the area per point is approximated only for the defaults.
        # this may not correctly generate points for other parameter values
        area_per_point = 19.938425320054257 / self.N
        cell_radius = np.sqrt(area_per_point / np.pi)
        self.const_kernel = ConstRepulsionKernel(cell_radius)
        self.repulsion_kernel = GaussianRepulsionKernel(height=1, shape=2 * cell_radius)

        if auto_settle:
            self.auto_settle(verbose=verbose, **tqdm_kwargs)

    def implicit_surf(self, points: np.ndarray[float]) -> np.ndarray[float]:
        x, y, z = points.T
        a, b, c, d = self.a, self.b, self.c, self.d
        return (
            (x**2 + y**2 + z**2 + b**2 - d**2) ** 2
            - 4 * (a * x - c * d) ** 2
            - 4 * b**2 * y**2
        )

    def implicit_surf_grad(self, points: np.ndarray[float]) -> np.ndarray[float]:
        x, y, z = points.T
        a, b, c, d = self.a, self.b, self.c, self.d
        ret = np.zeros_like(points)
        square = x**2 + y**2 + z**2 + b**2 - d**2
        ret[:, 0] = square * 4 * x - 8 * (a * x - c * d) * a
        ret[:, 1] = square * 4 * y - 8 * b**2 * y
        ret[:, 2] = square * 4 * z
        return ret

    def projection(
        self,
        points: np.ndarray[float],
        tol: float = 1e-10,
        iterations: int = 100,
        rate: float = 0.1,
    ) -> np.ndarray[float]:
        x = points + 0.0
        for _ in range(iterations):
            magnitude = self.implicit_surf(x)
            mask = np.abs(magnitude) > tol
            if not np.any(mask):
                break
            direction = self.implicit_surf_grad(x[mask])
            direction /= la.norm(direction, axis=1)[:, np.newaxis]
            x[mask] -= rate * magnitude[mask][:, np.newaxis] * direction
        return x

    @property
    def coords(self) -> tuple[np.ndarray[float]]:
        return (*self.points.T,)

    @property
    def normals(self) -> np.ndarray[float]:
        normals = self.implicit_surf_grad(self.points)
        normals /= la.norm(normals, axis=1)[:, np.newaxis]
        return normals

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
                projection=self.projection,
            )

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
                projection=self.projection,
            )

    def auto_settle(self, verbose: bool = False, **tqdm_kwargs):
        self.jostle(5_000, verbose=verbose, tqdm_kwargs=tqdm_kwargs)
        self.settle(0.005, repeat=100, verbose=verbose, tqdm_kwargs=tqdm_kwargs)


class SpiralCyclide(DupinCyclide):
    def __init__(self, N: int):
        torus = SpiralTorus(N, num_wraps=10)
        torus_points = torus.points
        us = np.arctan2(torus_points[:, 1], torus_points[:, 0])
        vs = np.arctan2(
            torus_points[:, 2],
            la.norm(torus_points[:, :2], axis=1) - torus.R,
        )
        a = 1
        b = 0.98
        c = 0.1983
        d = 0.5
        points = DupinCyclide.parameteric(us, vs, a=a, b=b, c=c, d=d)
        self.N = len(points)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        PointCloud.__init__(
            self,
            all_points=points,
            num_fixed=0,
            num_ghost=0,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pyvista as pv

    N = 1_000
    # shape = SpherePoints(N, auto_settle=False)
    shape = DupinCyclide(N, auto_settle=False)
    plt.ion()
    ax = plt.figure().add_subplot(projection="3d")
    (scatter,) = ax.plot(*shape.coords, "k.")

    for _ in tqdm(range(50)):
        shape.jostle(repeat=1, verbose=False)
        scatter.set_data_3d(*shape.coords)
        plt.pause(0.1)

    for _ in tqdm(range(100)):
        shape.settle(rate=1, repeat=1, verbose=False)
        scatter.set_data_3d(*shape.coords)
        plt.pause(0.1)

    vor = LocalSurfaceVoronoi(shape.points, shape.normals, shape.implicit_surf)

    mesh = pv.PolyData(shape.points, [(3, *f) for f in vor.triangles])

    # mesh.subdivide(1, inplace=True)
    mesh.subdivide_adaptive(inplace=True)
    mesh.points = shape.projection(mesh.points)
    print(f"N = {len(mesh.points)}")

    print(f"{np.max(np.abs(shape.implicit_surf(shape.points)))=}")
    plotter = pv.Plotter()
    # plotter.add_points(pv.PolyData(shape.points))
    plotter.add_mesh(
        mesh,
        show_edges=True,
    )
    plotter.show()
