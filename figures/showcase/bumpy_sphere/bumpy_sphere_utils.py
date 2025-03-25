import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree


def sphere_to_bumpy_sphere(
    points: np.ndarray[float],
    bump_centers: np.ndarray[float],
    bump_amplitude: float,
    bump_sd: float,
) -> np.ndarray[float]:
    new_points = points.copy()
    radii = 1 + bump_amplitude * sum(
        np.exp(-la.norm(points - center, axis=1) ** 2 / (2 * bump_sd**2))
        for center in bump_centers
    )
    new_points *= radii[:, np.newaxis]
    return new_points


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


def rotation_matrix(point: np.ndarray[float]) -> np.ndarray[float]:
    """
    Get a matrix that rotates the given point to the z-axis.
    """
    x, y, z = point / la.norm(point)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2)) - np.pi / 2
    R1 = np.array(
        [
            [np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    R2 = np.array(
        [
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ]
    )
    return R2 @ R1
