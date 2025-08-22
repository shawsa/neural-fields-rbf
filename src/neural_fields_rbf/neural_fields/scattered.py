from itertools import product
import numpy as np
import numpy.linalg as la
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare
from scipy.sparse import csr_array
from tqdm import tqdm


def euclidian_dist(points, x):
    return la.norm(points - x, axis=1)


class FlatTorrusDistance:
    def __init__(self, x_width, y_width):
        self.x_width = x_width
        self.y_width = y_width

    def __call__(self, points, x):
        x_offsets = [i * self.x_width for i in [-1, 0, 1]]
        y_offsets = [i * self.y_width for i in [-1, 0, 1]]
        dist_tensor = np.zeros((len(x_offsets) * len(y_offsets), len(points)))
        for index, (x_offset, y_offset) in enumerate(product(x_offsets, y_offsets)):
            x2 = x + np.array([x_offset, y_offset])
            dist_tensor[index] = la.norm(points - x2, axis=1)
        return np.min(dist_tensor, axis=0)


class NeuralField:
    def __init__(
        self,
        verbose=False,
        tqdm_kwargs={},
        dist=euclidian_dist,
        *,
        qf: LocalQuad,
        firing_rate: Callable[[np.ndarray], np.ndarray],
        weight_kernel: Callable[[np.ndarray], np.ndarray],
    ):
        self.qf = qf
        self.points = qf.points
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.dist = dist
        self.initialize_convolution(verbose=verbose, tqdm_kwargs=tqdm_kwargs)

    def initialize_convolution(self, verbose: bool, tqdm_kwargs):
        conv_mat = np.zeros((len(self.points), len(self.points)))
        verbose_wrapper = enumerate(self.points)
        if verbose:
            verbose_wrapper = tqdm(
                verbose_wrapper, total=len(self.points), **tqdm_kwargs
            )
        for index, point in verbose_wrapper:
            conv_mat[index] = self.qf.weights * self.weight_kernel(
                self.dist(self.points, point)
            )
        self.conv_mat = conv_mat

    def conv(self, arr: np.ndarray):
        return self.conv_mat @ arr

    def rhs(self, t, u):
        return -u + self.conv(self.firing_rate(u))


class NeuralFieldCustomKernel(NeuralField):
    def __init__(
        self,
        verbose=False,
        tqdm_kwargs={},
        dist=euclidian_dist,
        *,
        qf: LocalQuad,
        firing_rate: Callable[[np.ndarray], np.ndarray],
        weight_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.qf = qf
        self.points = qf.points
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.dist = dist
        self.initialize_convolution(verbose=verbose, tqdm_kwargs=tqdm_kwargs)

    def initialize_convolution(self, verbose: bool, tqdm_kwargs):
        conv_mat = np.zeros((len(self.points), len(self.points)))
        verbose_wrapper = enumerate(self.points)
        if verbose:
            verbose_wrapper = tqdm(
                verbose_wrapper, total=len(self.points), **tqdm_kwargs
            )
        for row, point in verbose_wrapper:
            for col, point2 in enumerate(self.points):
                conv_mat[row][col] = self.weight_kernel(point, point2)
            conv_mat[row] *= self.qf.weights
        self.conv_mat = conv_mat


class NeuralFieldSparse(NeuralField):
    def __init__(
        self,
        *,
        sparcity_tolerance: float = 1e-16,
        **kwargs,
    ):
        self.sparcity_tolerance = sparcity_tolerance
        super().__init__(**kwargs)

    def initialize_convolution(self, verbose: bool = False, tqdm_kwargs={}):
        data = []
        indices = []
        indptr = [0]
        shape = 2 * (len(self.points),)
        index_arr = np.arange(len(self.points), dtype=int)
        verbose_wrapper = self.points
        if verbose:
            verbose_wrapper = tqdm(
                verbose_wrapper, total=len(self.points), **tqdm_kwargs
            )
        for point in verbose_wrapper:
            row = self.qf.weights * self.weight_kernel(self.dist(self.points, point))
            row_mask = np.abs(row) > self.sparcity_tolerance
            data += list(row[row_mask])
            indices += list(index_arr[row_mask])
            indptr.append(len(data))

        self.conv_mat = csr_array((data, indices, indptr), shape=shape)
        self.fill = 100 * self.conv_mat.nnz / len(self.points) ** 2


class NeuralFieldMemMin(NeuralField):
    def initialize_convolution(self, verbose: bool = False, tqdm_kwargs={}):
        pass

    def conv(self, arr: np.ndarray):
        ret = np.empty_like(arr)
        for index, x in enumerate(self.points):
            ret[index] = self.qf.weights @ (
                self.weight_kernel(self.dist(self.points, x)) * arr
            )
        return ret
