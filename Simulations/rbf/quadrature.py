import numpy as np
import numpy.linalg as la
from rbf.geometry import Triangle, triangle
from rbf.poly_utils import poly_powers_gen
from rbf.quad_lib import get_right_triangle_integral_function
from rbf.rbf import RBF
from rbf.stencil import Stencil
from scipy.spatial import Delaunay, KDTree
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial


class QuadStencil(Stencil):
    """An object to store a quadrature stencil for a given Trianglular domain,
    and to compute the associated weights."""

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
    """Similar to QuadStencil, but also keeps track of global collocation indices.
    Useful for local quadrature."""
    def __init__(
        self, points: np.ndarray[float], element: Triangle, mesh_indices=np.ndarray[int]
    ):
        super(LocalQuadStencil, self).__init__(points, element=element)
        self.mesh_indices = mesh_indices


class LocalQuad:
    """An object to get quadrature weights by performing local RBF quadrature
    on a Delaunay triangulation of the quadrature nodes."""
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
        self.generate_weights()

    def initialize_mesh(self):
        self.mesh = Delaunay(self.points)

    @property
    def elements(self):
        return list(map(triangle, self.mesh.points[self.mesh.simplices]))

    def generate_weights(self):
        self.stencils = []
        self.weights = np.zeros(len(self.points))
        if self.verbose:
            wrapper = lambda gen: tqdm(gen, **self.tqdm_kwargs)
        else:
            wrapper = lambda gen: gen

        for element in wrapper(self.elements):
            _, neighbor_indices = self.kdt.query(element.centroid, self.stencil_size)
            stencil = LocalQuadStencil(
                self.points[neighbor_indices],
                element,
                neighbor_indices,
            )
            self.stencils.append(stencil)
            self.weights[stencil.mesh_indices] += stencil.weights(
                self.rbf, self.poly_deg
            )
