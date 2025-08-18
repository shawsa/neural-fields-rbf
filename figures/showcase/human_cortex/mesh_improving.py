import numpy as np
import numpy.linalg as la
import pickle
import pyvista as pv
from tqdm import tqdm

from rbf.surface import TriMesh
from mesh_repair_utils import MeshRepairProcess

OLD_MESH_FILE = "data/left_hemi_mesh.pickle"
NEW_MESH_FILE = "data/left_hemi_mesh_improved.pickle"

with open(OLD_MESH_FILE, "rb") as f:
    mesh_dict = pickle.load(f)

mesh = pv.PolyData(mesh_dict["vertices"], [(3, *f) for f in mesh_dict["triangles"]])
mesh = MeshRepairProcess(mesh)
mesh.repair()
mesh.subdivide(3)
mesh.cluster(20_000)
mesh.reduce(200_000)
mesh.cluster(2_000)

backup = mesh.mesh.copy()

if False:
    mesh = MeshRepairProcess(backup)

for num_neighbors in [20] * 100:
    changed_indices, old_points = mesh.edge_smooth(num_neighbors, angle=60.0)

for num_neighbors in [200] * 1:
    changed_indices, old_points = mesh.edge_smooth(num_neighbors, angle=60.0)

for num_neighbors in [20] * 5:
    changed_indices, old_points = mesh.edge_smooth(num_neighbors, angle=60.0)

mesh.smooth_taubin()

for num_neighbors in [20] * 5:
    changed_indices, old_points = mesh.edge_smooth(num_neighbors, angle=60.0)

mesh.smooth()

mesh.cluster(2_000)

mesh.smooth()

for num_neighbors in [50] * 5:
    changed_indices, old_points = mesh.edge_smooth(num_neighbors, angle=60.0)

mesh.smooth()

mesh.cluster(2_000)

for _ in range(10):
    changed_indices, old_points = mesh.edge_smooth(6, angle=75.0, n_iter=100)

mesh.cluster(2_000)

changed_indices, old_points = mesh.edge_smooth(
    50, angle=75.0, n_iter=1000, relaxation_factor=1.0
)

changed_indices, old_points = mesh.edge_smooth(
    50, angle=75.0, n_iter=1000, relaxation_factor=1.0
)

changed_indices, old_points = mesh.edge_smooth(
    50, angle=75.0, n_iter=1000, relaxation_factor=1.0
)

changed_indices, old_points = mesh.edge_smooth(
    200, angle=75.0, n_iter=1000, relaxation_factor=1.0
)

for _ in range(9):
    changed_indices, old_points = mesh.edge_smooth(
        200, angle=75.0, n_iter=1000, relaxation_factor=1.0
    )

mesh.cluster(2_000)

mesh.cluster(20_000)

feature_edges = mesh.mesh.extract_feature_edges(feature_angle=75, progress_bar=True)
print(feature_edges)
plotter = pv.Plotter()
plotter.add_mesh(
    mesh.mesh,
    show_edges=True,
    opacity=0.9,
)
plotter.add_mesh(feature_edges, color="red", line_width=10)
plotter.show()

plotter = pv.Plotter()
plotter.add_mesh(
    mesh.mesh,
    show_edges=True,
)
plotter.show()

mesh.mesh.compute_normals(point_normals=True, progress_bar=True)

trimesh = TriMesh(
    mesh.mesh.points,
    mesh.mesh.faces.reshape((-1, 4))[:, 1:],
    mesh.mesh.points,
)

normals = np.zeros_like(trimesh.points)
for index, vert in enumerate(tqdm(trimesh.points)):
    faces = [trimesh.face(face_id).index for face_id in trimesh.vertex_map[index]]
    normals[index] = np.sum(mesh.mesh.face_normals[np.array(faces, dtype=int)], axis=0)
normals /= la.norm(normals, axis=1)[:, np.newaxis]

with open(NEW_MESH_FILE, "wb") as f:
    pickle.dump(
        {
            "vertices": mesh.mesh.points,
            "triangles": mesh.mesh.faces.reshape((-1, 4))[:, 1:],
            "normals": mesh.mesh.point_normals,
        },
        f,
    )
