import numpy as np
import numpy.linalg as la


def rotation_matrix(a: np.ndarray[float], b: np.ndarray[float]) -> np.ndarray[float]:
    """Create a matrix that rotates the vector a to the vector b."""
    v = a/la.norm(a) + b/la.norm(b)
    R = 2/np.dot(v, v) * np.outer(v, v) - np.eye(3)
    return R


def _rotation_matrix2(a: np.ndarray[float]) -> np.ndarray[float]:
    nx, ny, nz = a / la.norm(a)
    proj_mag = np.sqrt(nx**2 + ny**2)
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    prod = proj_mag * mag
    return np.array([
        [nx*nz / prod, ny*nz / prod, -proj_mag / mag],
        [-ny/proj_mag, nx/proj_mag, 0],
        [nx / mag, ny / mag, nz/mag],
    ])


class Face:
    def __init__(self, index: int, trimesh, normal_orientaion=None):
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

    def neighbors(self):
        neighbors = [None, None, None]
        i, j, k = self.vert_indices
        for face_index, indices in enumerate(self.trimesh.simplices):
            if i in indices and j in indices:
                neighbors[0] = Face(
                    face_index, self.trimesh, normal_orientaion=self.normal
                )
            elif j in indices and k in indices:
                neighbors[1] = Face(
                    face_index, self.trimesh, normal_orientaion=self.normal
                )
            elif k in indices and i in indices:
                neighbors[2] = Face(
                    face_index, self.trimesh, normal_orientaion=self.normal
                )
        return neighbors


class TriMesh:
    """A wrapper class for a triangular mesh of a surface."""

    def __init__(self, points: np.ndarray[float], simplices: np.ndarray[int]):
        self.points = points
        self.simplices = simplices

    def get_face(self, face_index):
        return Face(face_index, self)

    @property
    def faces(self):
        for face_index in range(len(self.simplices)):
            yield self.get_face(face_index)

    def edge_normals(self, face, neighbors: list[Face]) -> list[np.ndarray[float]]:
        return [(face.normal + nbr.normal) / 2 for nbr in neighbors]

    def projection_point(self, face_index):
        face = self.get_face(face_index)
        neighbors = face.neighbors()
        nab, nbc, nca = self.edge_normals(face, neighbors)
        noab = np.cross(nab, face.b - face.a)
        nobc = np.cross(nbc, face.c - face.b)
        noca = np.cross(nca, face.a - face.c)
        voa = np.cross(noab, noca)
        return face.a + np.dot(nobc, face.b - face.a) / np.dot(nobc, voa) * voa
