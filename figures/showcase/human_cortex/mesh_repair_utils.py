import numpy as np
import pyvista as pv
import pyacvd
import pymeshfix

from scipy.spatial import KDTree
from rbf.surface import TriMesh


class MeshRepairProcess:
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh.triangulate()

    def repair(self):
        mesh_fix = pymeshfix.MeshFix(self.mesh)
        mesh_fix.repair(verbose=True)
        self.mesh = mesh_fix.mesh

    def smooth_taubin(self, n_iter=20):
        self.mesh.smooth_taubin(
            n_iter=n_iter,
            feature_smoothing=True,
            edge_angle=1000,
            feature_angle=1000,
            progress_bar=True,
            inplace=True,
        )

    def smooth(self, relaxation_factor=0.01, n_iter=200):
        self.mesh.smooth(
            relaxation_factor=relaxation_factor,
            n_iter=n_iter,
            feature_smoothing=True,
            edge_angle=1000,
            feature_angle=1000,
            progress_bar=True,
            inplace=True,
        )

    def subdivide(self, n: int):
        self.mesh.subdivide(n, progress_bar=True, inplace=True)

    def cluster(self, N: int):
        print(f"Clustering {N=}")
        clustering = pyacvd.Clustering(self.mesh)
        clustering.cluster(N)
        self.mesh = clustering.mesh

    def plot(self):
        self.mesh.plot(show_edges=True)

    def reduce(self, approx_N: int):
        target_reduction = 1 - approx_N / len(self.mesh.points)
        self.mesh.decimate(
            target_reduction,
            inplace=True,
            progress_bar=True,
        )

    def smooth_near_points(
        self,
        points: pv.PolyData,
        num_neighbors: int,
        relaxation_factor: float,
        n_iter: int,
    ):
        tree = KDTree(self.mesh.points)
        _, sub_mesh_indices = tree.query(points, k=num_neighbors)
        sub_mesh_indices = np.unique(np.ravel(sub_mesh_indices))
        print(f"adjusting {len(sub_mesh_indices)} points")
        sub_mesh = self.mesh.extract_points(sub_mesh_indices).extract_surface()
        _, sub_mesh_indices = tree.query(sub_mesh.points, k=1)
        old_points = sub_mesh.points
        sub_mesh.smooth(
            relaxation_factor=relaxation_factor,
            n_iter=n_iter,
            boundary_smoothing=False,
            feature_angle=1000,
            edge_angle=1000,
            progress_bar=True,
            inplace=True,
        )
        self.mesh.points[sub_mesh_indices] = sub_mesh.points
        return sub_mesh_indices, old_points

    def edge_smooth(
        self,
        num_neighbors: int,
        angle: float = 30.0,
        relaxation_factor: float = 1.0,
        n_iter: int = 1000,
    ):
        feature_edges = self.mesh.extract_feature_edges(
            feature_angle=angle, progress_bar=True
        )
        print(f"found {len(feature_edges.points)} points")
        return self.smooth_near_points(
            points=feature_edges.points,
            num_neighbors=num_neighbors,
            relaxation_factor=relaxation_factor,
            n_iter=n_iter,
        )

    def short_edge_smooth(
        self,
        num_neighbors: int,
        length_ratio: float,
        relaxation_factor: float = 1.0,
        n_iter: int = 1000,
    ):
        tree = KDTree(self.mesh.points)
        distances, indices = tree.query(self.mesh.points, k=2)
        distances = distances[:, 1]
        middle_value = np.median(distances)
        mask = distances < middle_value * length_ratio
        small_indices = np.unique(indices[mask, :].flatten())

        return self.smooth_near_points(
            points=self.mesh.points[small_indices],
            num_neighbors=num_neighbors,
            relaxation_factor=relaxation_factor,
            n_iter=n_iter,
        )


def subdivide_single_cell(trimesh: TriMesh, face_index: int) -> TriMesh:
    """Subdivide the target cell by bisecting its edges,
    and replacing the cell with four new triangles that cover the same area.
    Neighboring triangles are split into two new triangles each that
    cover the same area.

    This preserves the ordering of existing triangles and points.

    This results in three new vertices.
    This results in ten new triangles but removes four for a net gain of 6 triangles.
    """
    num_vert = len(trimesh.points)
    num_tri = len(trimesh.simplices)
    face = trimesh.face(face_index)
    points = np.empty((num_vert + 3, 3), dtype=float)
    points[:num_vert] = trimesh.points
    points[num_vert + 0] = 0.5 * (face.a + face.b)
    points[num_vert + 1] = 0.5 * (face.b + face.c)
    points[num_vert + 2] = 0.5 * (face.c + face.a)

    removed_indices = [face.index] + [nbr.index for nbr in face.neighbors]
    old_mask = np.ones(num_tri, dtype=bool)
    for rm in removed_indices:
        old_mask[rm] = False
    triangles = np.empty((num_tri + 6, 3), dtype=int)
    triangles[:num_tri][old_mask] = trimesh.simplices[old_mask]
    new_mask = np.zeros(num_tri + 6, dtype=bool)
    new_mask[:num_tri][~old_mask] = True
    new_mask[-6:] = True

    a_index, b_index, c_index = face.vert_indices
    for nbr in face.neighbors:
        if a_index in nbr.vert_indices and b_index in nbr.vert_indices:
            ab_new = [
                index
                for index in nbr.vert_indices
                if index != a_index and index != b_index
            ][0]
            ab_normal = nbr.normal
        elif b_index in nbr.vert_indices and c_index in nbr.vert_indices:
            bc_new = [
                index
                for index in nbr.vert_indices
                if index != b_index and index != c_index
            ][0]
            bc_normal = nbr.normal
        elif a_index in nbr.vert_indices and c_index in nbr.vert_indices:
            ac_new = [
                index
                for index in nbr.vert_indices
                if index != a_index and index != c_index
            ][0]
            ac_normal = nbr.normal

    triangles[new_mask] = np.array(
        [
            [num_vert + 0, num_vert + 1, num_vert + 2],  # middle center
            [num_vert + 0, num_vert + 2, a_index],  # middle a
            [num_vert + 0, num_vert + 1, b_index],  # middle b
            [num_vert + 1, num_vert + 2, c_index],  # middle c
            [num_vert, a_index, ab_new],  # side ab split
            [num_vert, b_index, ab_new],  # side ab split
            [num_vert + 1, b_index, bc_new],  # side bc split
            [num_vert + 1, c_index, bc_new],  # side bc split
            [num_vert + 2, a_index, ac_new],  # side ac split
            [num_vert + 2, c_index, ac_new],  # side ac split
        ],
        dtype=int,
    )

    normals = np.empty_like(points)
    normals[:num_vert] = trimesh.normals
    normals[num_vert : num_vert + 4] = face.normal[np.newaxis, :]
    normals[num_vert + 4 : num_vert + 6] = ab_normal[np.newaxis, :]
    normals[num_vert + 6 : num_vert + 8] = bc_normal[np.newaxis, :]
    normals[num_vert + 8 :] = ac_normal[np.newaxis, :]

    return TriMesh(points, triangles, normals)
