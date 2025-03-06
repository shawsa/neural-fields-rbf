from collections import defaultdict
from functools import cache
import numpy as np
import numpy.linalg as la
from tqdm import tqdm

from rbf.geometry import triangle
from rbf.rbf import RBF
from rbf.quadrature import QuadStencil, LocalQuadStencil
from scipy.spatial import KDTree


def rotation_matrix(a: np.ndarray[float], b: np.ndarray[float]) -> np.ndarray[float]:
    """Create a matrix that rotates the vector a to the vector b."""
    v = a / la.norm(a) + b / la.norm(b)
    d = np.dot(v, v)
    if d < 1e-14:
        return np.eye(3)
    R = 2 / np.dot(v, v) * np.outer(v, v) - np.eye(3)
    return R


class TriMesh:
    """A wrapper class for a triangular mesh of a surface."""

    def __init__(
        self,
        points: np.ndarray[float],
        simplices: np.ndarray[int],
        normals: np.ndarray[float],
    ):
        self.points = points
        self.simplices = simplices
        self.normals = normals
        self.init_vertex_map()

    def face(self, face_index):
        corner_index = self.simplices[face_index][0]
        return Face(face_index, self, self.normals[corner_index])

    @property
    def faces(self):
        for face_index in range(len(self.simplices)):
            yield self.face(face_index)

    def init_vertex_map(self):
        self.vertex_map = [set() for _ in range(len(self.points))]
        for face_index, simplices in enumerate(self.simplices):
            for index in simplices:
                self.vertex_map[index].add(face_index)

    def neighbors(self, face_index):
        i, j, k = self.simplices[face_index]
        i_set = self.vertex_map[i]
        j_set = self.vertex_map[j]
        k_set = self.vertex_map[k]
        neighbors = [
            index for index in i_set.intersection(j_set) if index != face_index
        ]
        neighbors += [
            index for index in j_set.intersection(k_set) if index != face_index
        ]
        neighbors += [
            index for index in k_set.intersection(i_set) if index != face_index
        ]
        return neighbors

    def is_valid(self):
        return all(len(self.neighbors(i)) == 3 for i in range(len(self.simplices)))


class Face:
    def __init__(self, index: int, trimesh: TriMesh, normal_orientaion=None):
        self.index = index
        self.trimesh = trimesh
        self.calculate_normal(normal_orientation=normal_orientaion)

    @property
    def vert_indices(self) -> np.ndarray[int]:
        return self.trimesh.simplices[self.index]

    @property
    def verts(self) -> np.ndarray[float]:
        return self.trimesh.points[self.vert_indices]

    def __repr__(self):
        return f"{self.verts}"

    @property
    def a(self) -> np.ndarray[float]:
        return self.verts[0]

    @property
    def b(self) -> np.ndarray[float]:
        return self.verts[1]

    @property
    def c(self) -> np.ndarray[float]:
        return self.verts[2]

    @property
    def center(self) -> np.ndarray[float]:
        return (self.a + self.b + self.c) / 3

    def calculate_normal(self, normal_orientation):
        normal = np.cross(self.b - self.a, self.c - self.a)
        if normal_orientation is None:
            normal_orientation = self.center
        sign = np.sign(np.dot(normal, normal_orientation))
        self.normal = normal * sign / la.norm(normal)

    @property
    def neighbors(self):
        return [
            Face(index, self.trimesh, self.normal)
            for index in self.trimesh.neighbors(self.index)
        ]

    @property
    def edge_normals(self) -> list[np.ndarray[float]]:
        return [(self.normal + nbr.normal) / 2 for nbr in self.neighbors]

    @property
    @cache
    def projection_point(self):
        nab, nbc, nca = self.edge_normals
        noab = np.cross(nab, self.b - self.a)
        nobc = np.cross(nbc, self.c - self.b)
        noca = np.cross(nca, self.a - self.c)
        voa = np.cross(noab, noca)
        return self.a + np.dot(nobc, self.b - self.a) / np.dot(nobc, voa) * voa


class SurfaceStencil(LocalQuadStencil):
    def __init__(
        self,
        face: Face,
        points: np.ndarray[float],
        normals: np.ndarray[float],
        point_indices: np.ndarray[int],
    ):
        self.face = face
        self.points = points
        self.normals = normals
        self.point_indices = point_indices

    @property
    def rotation_matrix(self) -> np.ndarray[float]:
        return rotation_matrix(self.face.normal, np.array([0, 0, 1]))

    def gnomic_proj(self, points) -> np.ndarray[float]:
        normal = self.face.normal
        center = self.face.center - self.face.projection_point
        points = points - self.face.projection_point
        return (
            np.cross(normal, np.cross(points, center))
            / np.dot(points, normal)[:, np.newaxis]
        ) @ self.rotation_matrix[:2].T

    @property
    def planar_points(self) -> np.ndarray[float]:
        return self.gnomic_proj(self.points)

    @property
    def planar_face_verts(self) -> np.ndarray[float]:
        return self.gnomic_proj(self.face.verts)

    def flat_weights(self, rbf: RBF, poly_deg: int):
        return QuadStencil(
            self.planar_points, triangle(self.planar_face_verts)
        ).weights(rbf, poly_deg)

    def weights(self, rbf: RBF, poly_deg: int):
        flat_weights = self.flat_weights(rbf, poly_deg)
        weights = np.zeros_like(flat_weights)
        proj = self.face.projection_point
        # make faster?
        for index, (w, pnt, normal) in enumerate(
            zip(flat_weights, self.points, self.normals)
        ):
            ref = pnt - proj
            num = np.dot(ref, self.face.normal)
            weights[index] = (
                w
                * num
                / np.dot(ref, normal)
                * (num / np.dot(self.face.normal, self.face.a - proj)) ** 2
            )

        return weights


class SurfaceQuad:
    """Generate quadrature weights for vertices of a triangular mesh of a closed 2D
    surface embedded in 3D space.

    Uses the algorithm from Reeger JA, Fornberg B, Watts ML. 2016
    Numerical quadrature over smooth, closed surfaces. Proc.R.Soc.A472: 20160401.
    http://dx.doi.org/10.1098/rspa.2016.0401
    """

    def __init__(
        self,
        trimesh: TriMesh,
        rbf: RBF,
        poly_deg: int,
        stencil_size: int,
        verbose=False,
        tqdm_kwargs={},
    ):
        self.trimesh = trimesh
        self.rbf = rbf
        self.poly_deg = poly_deg
        self.stencil_size = stencil_size
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs

        self.kdt = KDTree(self.trimesh.points)
        self.generate_weights()

    @property
    def points(self):
        return self.trimesh.points

    def generate_weights(self):
        self.stencils = []
        self.weights = np.zeros(len(self.trimesh.points))
        if self.verbose:

            def wrapper(gen):
                return tqdm(gen, total=len(self.trimesh.simplices), **self.tqdm_kwargs)

        else:

            def wrapper(gen, **_):
                return gen

        for face in wrapper(self.trimesh.faces):
            _, neighbor_indices = self.kdt.query(face.center, self.stencil_size)
            stencil = SurfaceStencil(
                face,
                self.trimesh.points[neighbor_indices],
                self.trimesh.normals[neighbor_indices],
                neighbor_indices,
            )
            self.stencils.append(stencil)
            self.weights[neighbor_indices] += stencil.weights(self.rbf, self.poly_deg)
