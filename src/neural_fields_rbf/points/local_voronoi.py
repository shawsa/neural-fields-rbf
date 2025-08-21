import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree
from tqdm import tqdm
from typing import Callable


class Cell:
    def __init__(
        self,
        center: np.ndarray[float],
        tree: KDTree,
        max_neighbors: int = 20,
        bnd_normals: np.ndarray[float] = [],
        bnd_offsets: np.ndarray[float] = [],
    ):
        assert len(bnd_normals) == len(bnd_offsets)
        self.center = center
        _, nbid = tree.query(center, k=max_neighbors + 1)
        nbid = nbid[1:]
        neighbors = tree.data[nbid]

        normals = self._get_normals(center, neighbors, bnd_normals)
        offsets = self._get_offsets(center, neighbors, bnd_offsets)

        index1, index2, self.vertices = self._find_vertices(normals, offsets)
        mask = np.logical_and(index1 < max_neighbors, index2 < max_neighbors)
        index1 = index1[mask]
        index2 = index2[mask]

        self.neighbor_ids = nbid[
            np.array(list(set(index1).union(set(index2))), dtype=int)
        ]
        self.vert_bnd1_ids = nbid[index1]
        self.vert_bnd2_ids = nbid[index2]

    def _non_paraellel_pairs(
        self,
        normals: np.ndarray[float],
        tol=1e-14,
    ) -> np.ndarray[bool]:
        mask = np.abs(normals @ normals.T) < 1 - tol
        return mask

    def _contained_mat(
        self,
        points: np.ndarray[float],
        normals: np.ndarray[float],
        offsets: np.ndarray[float],
        tol=1e-14,
    ):
        return (
            sum(
                normals[:, i] * np.subtract.outer(points[:, i], offsets[:, i])
                for i in range(normals.shape[-1])
            )
            > -tol
        )

    def _get_normals(
        self,
        center: np.ndarray[float],
        neighbors: np.ndarray[float],
        bnd_normals: np.ndarray[float],
    ) -> np.ndarray[float]:
        n = len(neighbors)
        normals = np.empty((n + len(bnd_normals), len(center)))
        normals[:n] = center - neighbors
        if len(bnd_normals) > 0:
            normals[n:] = bnd_normals
        normals /= la.norm(normals, axis=-1)[:, np.newaxis]
        return normals

    def _get_offsets(
        self,
        center: np.ndarray[float],
        neighbors: np.ndarray[float],
        bnd_offsets: np.ndarray[float],
    ) -> np.ndarray[float]:
        n = len(neighbors)
        offsets = np.empty((n + len(bnd_offsets), len(center)))
        offsets[:n] = (neighbors + center) / 2
        if len(bnd_offsets) > 0:
            offsets[n:] = bnd_offsets
        return offsets

    def _find_intersections(
        self,
        normals: np.ndarray[float],
        offsets: np.ndarray[float],
    ):
        index1, index2 = np.meshgrid(*2 * (range(len(normals)),), indexing="ij")
        # remove doulbe counting
        mask = np.triu(self._non_paraellel_pairs(normals), k=1)
        index1 = index1[mask]
        index2 = index2[mask]

        n1 = normals[index1]
        n2 = normals[index2]
        o1 = offsets[index1]
        o2 = offsets[index2]

        num_points = np.sum(mask)

        mat = np.zeros((num_points, 2, 2))
        mat[:, 0] = n1
        mat[:, 1] = n2

        rhs = np.zeros((num_points, 2))
        rhs[:, 0] = np.sum(n1 * o1, axis=1)
        rhs[:, 1] = np.sum(n2 * o2, axis=1)

        verts = la.solve(mat, rhs)
        return index1, index2, verts

    def _find_vertices(
        self,
        normals: np.ndarray[float],
        offsets: np.ndarray[float],
    ):
        index1, index2, verts = self._find_intersections(normals, offsets)
        mask = np.all(self._contained_mat(verts, normals, offsets), axis=1)
        return index1[mask], index2[mask], verts[mask]


class LocalVoronoi:
    def __init__(
        self,
        points: np.ndarray[float],
        tree: KDTree = None,
        boundary_normals: np.ndarray[float] = [],
        boundary_offsets: np.ndarray[float] = [],
        max_neighbors: int = 15,
        compute_circum_radius: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        self.points = points
        assert len(boundary_normals) == len(boundary_offsets)
        self.boundary_normals = boundary_normals
        self.boundary_offsets = boundary_offsets
        if tree is None:
            tree = KDTree(points)
        self.tree = tree
        self.max_neighbors = max_neighbors

        self.init_cells()
        self.init_triangles()
        if compute_circum_radius:
            self.init_circum_radius()
        else:
            self.circum_radius = None

    def init_cells(
        self,
    ):
        def null_wrapper(iterable):
            return iterable

        def verbose_wrapper(iterable):
            return tqdm(
                iterable,
                desc="Calculating Local Voronoi",
                total=len(self.points),
                **self.tqdm_kwargs,
            )

        tqdm_wrapper = null_wrapper
        if self.verbose:
            tqdm_wrapper = verbose_wrapper
        self.cells = [
            Cell(
                center,
                self.tree,
                self.max_neighbors,
                self.boundary_normals,
                self.boundary_offsets,
            )
            for index, center in tqdm_wrapper(enumerate(self.points))
        ]

    def init_triangles(self):
        triangles = set()
        for id1, cell in enumerate(self.cells):
            for id2, id3 in zip(cell.vert_bnd1_ids, cell.vert_bnd2_ids):
                triangles.add(tuple(sorted([id1, id2, id3])))
        self.triangles = np.array(list(triangles), dtype=int)

    def init_circum_radius(self):
        self.circum_radius = max(
            min(la.norm(cell.center - vert) for vert in cell.vertices)
            for cell in self.cells
        )


class SurfaceCell(Cell):
    def __init__(
        self,
        center: np.ndarray[float],
        center_normal: np.ndarray[float],
        implicit_surf: Callable[[np.ndarray[float]], np.ndarray[float]],
        tree: KDTree,
        max_neighbors: int = 20,
        bnd_normals: np.ndarray[float] = None,
        bnd_offsets: np.ndarray[float] = None,
    ):
        self.center_normal = center_normal
        self.implicit_surf = implicit_surf
        super().__init__(
            center=center,
            tree=tree,
            max_neighbors=max_neighbors,
            bnd_normals=bnd_normals,
            bnd_offsets=bnd_offsets,
        )

    def _find_intersections(
        self,
        normals: np.ndarray[float],
        offsets: np.ndarray[float],
    ):
        index1, index2 = np.meshgrid(*2 * (range(len(normals)),), indexing="ij")
        # remove doulbe counting
        mask = np.triu(self._non_paraellel_pairs(normals), k=1)
        index1 = index1[mask]
        index2 = index2[mask]

        n1 = normals[index1]
        n2 = normals[index2]
        o1 = offsets[index1]
        o2 = offsets[index2]

        num_points = np.sum(mask)

        mat = np.zeros((num_points, 3, 3))
        mat[:, 0] = n1
        mat[:, 1] = n2
        mat[:, 2] = self.center_normal

        rhs = np.zeros((num_points, 3))
        rhs[:, 0] = np.sum(n1 * o1, axis=1)
        rhs[:, 1] = np.sum(n2 * o2, axis=1)
        rhs[:, 2] = np.dot(self.center_normal, self.center)

        singular_mask = la.cond(mat) < 1e15
        mat = mat[singular_mask]
        rhs = rhs[singular_mask]
        index1 = index1[singular_mask]
        index2 = index2[singular_mask]
        n1 = n1[singular_mask]
        n2 = n2[singular_mask]

        approx_vert = la.solve(mat, rhs)
        cross = np.cross(n1, n2)
        # secant method to project onto surface

        def foo(t):
            return self.implicit_surf(approx_vert + cross * t[:, np.newaxis])

        t0, t1 = np.zeros(len(approx_vert)), 1e-5 * np.ones(len(approx_vert))
        mask = np.ones_like(t0, dtype=bool)
        f1 = foo(t0)
        for _ in range(20):
            f0, f1 = f1, foo(t1)
            mask = np.abs(f1 - f0) > 1e-14
            if not np.any(mask):
                break
            t0, t1[mask] = t1.copy(), t1[mask] - f1[mask] * (t0[mask] - t1[mask]) / (
                f0[mask] - f1[mask]
            )
        f1 = foo(t1)
        mask = np.abs(f1) < 1e-13
        verts = approx_vert[mask] + t1[mask][:, np.newaxis] * cross[mask]
        return index1[mask], index2[mask], verts


class LocalSurfaceVoronoi(LocalVoronoi):
    def __init__(
        self,
        points: np.ndarray[float],
        normals: np.ndarray[float],
        implicit_surf: Callable[[np.ndarray[float]], np.ndarray[float]],
        tree: KDTree = None,
        boundary_normals: np.ndarray[float] = [],
        boundary_offsets: np.ndarray[float] = [],
        max_neighbors: int = 15,
        compute_cirum_radius: bool = True,
        verbose: bool = False,
        tqdm_kwargs={},
    ):
        self.normals = normals
        self.implicit_surf = implicit_surf
        super().__init__(
            points=points,
            tree=tree,
            boundary_normals=boundary_normals,
            boundary_offsets=boundary_offsets,
            max_neighbors=max_neighbors,
            compute_circum_radius=compute_cirum_radius,
            verbose=verbose,
            tqdm_kwargs=tqdm_kwargs,
        )

    def init_cells(self):
        if self.verbose:

            def tqdm_wrapper(iterable):
                return tqdm(
                    iterable,
                    desc="Calculating Local Voronoi",
                    total=len(self.points),
                    **self.tqdm_kwargs,
                )

        else:

            def tqdm_wrapper(iterable):
                return iterable

        self.cells = [
            SurfaceCell(
                center=center,
                center_normal=self.normals[index],
                implicit_surf=self.implicit_surf,
                tree=self.tree,
                bnd_normals=self.boundary_normals,
                bnd_offsets=self.boundary_offsets,
                max_neighbors=self.max_neighbors,
            )
            for index, center in tqdm_wrapper(enumerate(self.points))
        ]


if __name__ == "__main__":
    from collections import defaultdict
    from itertools import combinations
    import matplotlib.pyplot as plt
    from min_energy_points.unit_square import UnitSquare
    from min_energy_points.torus import TorusPoints, SpiralTorus
    import pyvista as pv
    from scipy.spatial import Voronoi, voronoi_plot_2d

    np.random.seed(0)
    N = 100
    square = UnitSquare(N, verbose=True)
    points = square.points
    tree = KDTree(points)
    bnd_normals = np.array([bnd.normal for bnd in square.boundary_force.boundaries])
    bnd_offsets = np.array([bnd.offset for bnd in square.boundary_force.boundaries])

    vor = LocalVoronoi(
        points,
        tree=tree,
        boundary_normals=bnd_normals,
        boundary_offsets=bnd_offsets,
    )

    plt.figure("Square")
    plt.plot(*points.T, "k.")
    voronoi_plot_2d(Voronoi(points), ax=plt.gca())
    plt.triplot(*points.T, vor.triangles)

    for cell_approx in (
        np.r_[0.5, 0.5],
        np.r_[0.0, 0.5],
        np.r_[1.0, 1.0],
    ):
        _, center_index = tree.query(cell_approx, k=1)
        cell = vor.cells[center_index]

        plt.axis("equal")
        plt.plot(*cell.center, "g*", markersize=20)
        plt.plot(*points[cell.neighbor_ids].T, "bs")
        plt.plot(*cell.vertices.T, "rv")

    np.random.seed(0)
    N = 3_000
    torus = TorusPoints(N, verbose=True)
    # torus = SpiralTorus(N)
    # torus.settle(rate=1, repeat=10_000, verbose=True)
    vor = LocalSurfaceVoronoi(
        points=torus.points,
        normals=torus.normals,
        implicit_surf=torus.implicit_surf,
    )

    # find unattached triangles
    edge_count = defaultdict(lambda: 0)
    for tri in vor.triangles:
        for edge in combinations(tri, 2):
            key = tuple(sorted(edge))
            edge_count[key] += 1
    incomplete = set()
    for index, tri in enumerate(vor.triangles):
        for edge in combinations(tri, 2):
            if edge_count[key] != 2:
                incomplete.add(index)
    print(f"{len(incomplete)=}")

    points = torus.points
    surf = pv.PolyData(points, [(3, *f) for f in vor.triangles])
    # surf_normals = pv.PolyData(points)
    # surf_normals["vectors"] = torus.normals * 0.2
    # surf_normals.set_active_vectors("vectors")
    #
    # from itertools import combinations
    # max_dist = 0
    # for a, b, c in vor.triangles:
    #     for i, j in combinations([a, b, c], 2):
    #         dist = la.norm(points[i] - points[j])
    #         if dist > max_dist:
    #             max_dist = dist
    #             test_indices = sorted([a, b, c])

    # a, b, c = test_indices
    # cell = vor.cells[b]

    # verts = pv.PolyData(cell.vertices)
    # nbrs = pv.PolyData(points[cell.neighbor_ids])

    plotter = pv.Plotter(off_screen=False)
    plotter.add_mesh(
        surf,
        point_size=20,
        show_vertices=True,
        show_edges=True,
        cmap=["#AAAAAA", "#005500", "#774444"],
        show_scalar_bar=False,
    )
    # plotter.add_mesh(surf_normals.arrows, show_scalar_bar=False)
    # plotter.add_mesh(verts, color="k", point_size=10)
    # plotter.add_mesh(nbrs, color="r")
    plotter.show()
