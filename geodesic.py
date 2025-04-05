import meshlib.mrmeshpy as mesh
import meshlib.mrmeshnumpy as meshn
import numpy as np


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

    def dist(self, point: np.ndarray[float]) -> np.ndarray[float]:
        start_point = mesh.Vector3f(*point)
        my_dist = mesh.computeSurfaceDistances(
            self.triangulation,
            mesh.findProjection(
                start_point, self.triangulation
            ).mtp,
        )
        return np.array(my_dist.vec)
