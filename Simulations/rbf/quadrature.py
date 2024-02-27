import numpy as np
import numpy.linalg as la
from rbf.geometry import Triangle, triangle
from rbf.poly_utils import poly_powers_gen
from rbf.quad_lib import get_right_triangle_integral_function
from rbf.rbf import RBF
from rbf.stencil import Stencil
from scipy.spatial import Delaunay, KDTree
from tqdm import tqdm


class QuadStencil(Stencil):
    def __init__(self, points: np.ndarray[float], element: Triangle):
        super(QuadStencil, self).__init__(points, center=element.centroid)
        self.element = element
        self.scaled_element = (element - self.center) / self.scale_factor

    def weights(self, rbf: RBF, poly_deg: int):
        right_triangle_integrate = get_right_triangle_integral_function(rbf)
        mat = self.interpolation_matrix(rbf, poly_deg)
        rhs = np.zeros_like(mat[0])
        rhs[: len(self.points)] = self.scaled_element.rbf_quad(
            self.scaled_points, right_triangle_integrate
        )

        rhs[len(self.points) :] = np.array(
            [
                self.scaled_element.poly_quad(poly)
                for poly in poly_powers_gen(self.dim, poly_deg)
            ]
        )
        weights = la.solve(mat, rhs)
        return weights[: len(self.points)] * self.scale_factor**2


class LocalQuadStencil(QuadStencil):
    def __init__(
        self, points: np.ndarray[float], element: Triangle, mesh_indices=np.ndarray[int]
    ):
        super(LocalQuadStencil, self).__init__(points, element=element)
        self.mesh_indices = mesh_indices


class LocalQuad:
    def __init__(
        self,
        points: np.ndarray[float],
        rbf: RBF,
        poly_deg: int,
        stencil_size: int,
        verbose=False,
        tqdm_kwargs={},
    ):
        self.points = points
        self.rbf = rbf
        self.poly_deg = poly_deg
        self.stencil_size = stencil_size
        self.verbose = verbose
        self.tqdm_kwargs = tqdm_kwargs
        self.kdt = KDTree(self.points)
        self.initialize_mesh()
        self.initialize_stencils()
        self.generate_weights()

    def initialize_mesh(self):
        self.mesh = Delaunay(self.points)

    @property
    def elements(self):
        for tri_indices in self.mesh.simplices:
            yield triangle(self.mesh.points[tri_indices])

    def initialize_stencils(self):
        self.stencils = []
        for element in self.elements:
            _, neighbor_indices = self.kdt.query(element.centroid, self.stencil_size)
            self.stencils.append(
                LocalQuadStencil(
                    self.points[neighbor_indices],
                    element,
                    neighbor_indices,
                )
            )

    def generate_weights(self):
        if self.verbose:
            wrapper = lambda gen: tqdm(gen, **self.tqdm_kwargs)
        else:
            wrapper = lambda gen: gen
        self.weights = np.zeros(len(self.points))
        for stencil in wrapper(self.stencils):
            self.weights[stencil.mesh_indices] += stencil.weights(
                self.rbf, self.poly_deg
            )
