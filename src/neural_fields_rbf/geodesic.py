from collections import defaultdict
import meshlib.mrmeshpy as mesh
import meshlib.mrmeshnumpy as meshn
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix

from tqdm import tqdm


class GeodesicCalculator:
    """A container for some meshlib objects that we use to
    calculate geodesic distances.
    """

    def __init__(self, points: np.ndarray[float], triangles: np.ndarray[int] = None):
        self.points = points
        cloud = meshn.pointCloudFromPoints(points)
        cloud.invalidateCaches()
        if triangles is None:
            self.triangulation = mesh.triangulatePointCloud(cloud)
        else:
            self.triangulation = meshn.meshFromFacesVerts(triangles, points)

        self.edge_list = None

    def dist(self, point: np.ndarray[float]) -> np.ndarray[float]:
        start_point = mesh.Vector3f(*point)
        my_dist = mesh.computeSurfaceDistances(
            self.triangulation,
            mesh.findProjection(start_point, self.triangulation).mtp,
        )
        return np.array(my_dist.vec)

    def _generate_edge_list(self, verbose: bool = False):

        def do_nothing(x):
            return x

        wrapper = do_nothing

        triangles = [
            list(map(lambda vert: vert.get(), face))
            for face in wrapper(self.triangulation.topology.getTriangulation().vec)
        ]
        self.edge_list = defaultdict(set)
        for a, b, c in wrapper(triangles):
            self.edge_list[a].add(b)
            self.edge_list[a].add(c)
            self.edge_list[b].add(a)
            self.edge_list[b].add(c)
            self.edge_list[c].add(b)
            self.edge_list[c].add(a)

    def to_flat(
        self, center_index: int, num_hops: int, verbose: bool = False
    ) -> tuple[np.ndarray[float], np.ndarray[int]]:
        if self.edge_list is None:
            self._generate_edge_list(verbose=verbose)

        def do_nothing(x):
            return x

        wrapper = do_nothing
        if verbose:
            wrapper = tqdm

        if verbose:
            print("finding neighbors")
        my_indices = set()
        leaves = [center_index]
        for _ in wrapper(range(num_hops)):
            my_indices.update(leaves)
            leaves = [
                nbr
                for leaf in leaves
                for nbr in self.edge_list[leaf]
                if nbr not in my_indices
            ]
        my_indices.update(leaves)

        local_map = list(my_indices)
        n = len(local_map)

        if verbose:
            print("inverting map")
        local_inv = {val: key for key, val in enumerate(local_map)}

        dists = self.dist(self.points[center_index])
        dists /= np.max(dists)
        local_map.sort(key=lambda index: dists[index])
        dists = dists[local_map]

        if verbose:
            print("making local edge list")

        my_edge_list = {
            local_index: [
                local_inv[idx] for idx in self.edge_list[index] if idx in my_indices
            ]
            for local_index, index in wrapper(enumerate(local_map))
        }

        angles = np.random.random(dists.shape) * 2 * np.pi
        points = np.c_[dists * np.cos(angles), dists * np.sin(angles)]

        def normalize(points):
            norm = la.norm(points, axis=1)
            zero_mask = norm > 1e-10
            my_points = points.copy()
            my_points[zero_mask] /= norm[zero_mask][:, np.newaxis]
            return my_points

        if verbose:
            print(f"placing {n=} points")
        for rate in wrapper([1.0] * 1000 + [0.5] * 100 + [0.1]*100):
            grad = sum(
                normalize(point - points) * (np.exp(-(dists**2) * n/2))[:, np.newaxis]
                for point, dists in zip(points, distance_matrix(points, points))
            )
            # for index, point in enumerate(points):
            #     grad += sum(points[my_edge_list[index]] - point) / 50

            points += grad * rate
            points -= points[0]

        return points, local_map, my_edge_list
