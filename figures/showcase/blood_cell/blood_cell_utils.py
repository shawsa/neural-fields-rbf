import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree


def runge(rs: np.ndarray[float], amplitude: float, shape: float) -> np.ndarray[float]:
    return amplitude / (1 + (rs / shape) ** 2)


def flatten(
    points: np.ndarray[float], amplitude: float = 0.5, shape: float = 1
) -> np.ndarray[float]:
    points = points.copy()
    rs = la.norm(points[:, :2], axis=1)
    points[:, 2] *= 1 - runge(rs, amplitude, shape)
    return points


def unflatten(
    points: np.ndarray[float], amplitude: float = 0.5, shape: float = 1
) -> np.ndarray[float]:
    points = points.copy()
    rs = la.norm(points[:, :2], axis=1)
    points[:, 2] /= 1 - runge(rs, amplitude, shape)
    return points


def approx_normal(
    point: np.ndarray[float],
    points: np.ndarray[float],
    tree: KDTree,
    num_neighbors=12,
):
    """Approximate the normal vector to a point using the last principle component
    of the SVD of the closest points.
    """
    stencil = tree.query(point, k=num_neighbors)[1]
    pnts = points[stencil] - point
    normal = la.svd(pnts)[2][2]
    if np.dot(point, normal) < 0:
        normal *= -1
    return normal


def approximate_normal_vectors(
    points: np.ndarray[float],
    num_neighbors=12,
    tree: KDTree = None,
):
    if tree is None:
        tree = KDTree(points)
    normals = np.empty_like(points)
    for index, point in enumerate(points):
        normals[index] = approx_normal(point, points, tree, num_neighbors=num_neighbors)
    return normals
