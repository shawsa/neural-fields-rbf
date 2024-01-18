from functools import reduce
from itertools import product
import numpy as np
import operator


class Poly2D:
    def __init__(self, coeff_mat: np.ndarray[float]):
        """coeff_mat[m,n] denotes the coefficient on x^m*y^n"""
        self.C = coeff_mat

    @property
    def shape(self):
        return self.C.shape

    def reshape(self, shape: tuple[int]):
        assert all(dp <= dq for dp, dq in zip(self.shape, shape))
        new_mat = np.zeros(shape)
        new_mat[: self.shape[0], : self.shape[1]] = self.C
        return Poly2D(new_mat)

    def __repr__(self):
        n, m = self.shape
        pad = len(f"x^{m}y^{n}")
        ret = ""
        for row in range(n):
            for col, c in enumerate(self.C[row]):
                ret += f"\t{c:.3e}" + f"x^{row}y^{col}".ljust(pad)
            ret += "\n"
        return ret[:-1]

    def __call__(self, x, y):
        n, m = self.shape
        return sum(self.C[i, j] * x**i * y**j for j in range(m) for i in range(n))

    def __add__(self, q):
        if type(q) in [int, float]:
            ret = Poly2D(self.C.copy())
            ret.C[0, 0] += q
            return ret
        n1, m1 = self.shape
        n2, m2 = q.shape
        shape = (max(n1, n2), max(m1, m2))
        return Poly2D(self.reshape(shape).C + q.reshape(shape).C)

    def __radd__(self, q):
        return self + q

    def __mul__(self, q):
        if type(q) in [int, float]:
            return Poly2D(self.C * q)
        n1, m1 = self.shape
        n2, m2 = q.shape
        shape = (n1 + n2 - 1, m1 + m2 - 1)
        p = self.reshape(shape).C
        q = q.reshape(shape).C
        prod_mat = np.zeros(shape)
        for i, j in product(range(shape[0]), range(shape[1])):
            prod_mat[i, j] = np.sum(p[:i+1, :j+1][::-1, ::-1] * q[:i+1, :j+1])
        return Poly2D(prod_mat)

    def __rmul__(self, q):
        return self * q

    def __pow__(self, n: int):
        return reduce(operator.mul, [1] + [self]*n)

    def adiff_x(self):
        rows = self.shape[0] + 1
        cols = self.shape[1]
        shape = (rows, cols)
        mat = np.zeros(shape)
        for row in range(1, rows):
            mat[row] = self.C[row-1] / row
        return Poly2D(mat)

    def adiff_y(self):
        rows = self.shape[0]
        cols = self.shape[1] + 1
        shape = (rows, cols)
        mat = np.zeros(shape)
        for col in range(1, cols):
            mat[:, col] = self.C[:, col-1] / col
        return Poly2D(mat)


x = Poly2D(np.array([[0], [1]]))
y = Poly2D(np.array([[0, 1]]))

if __name__ == "__main__":
    p = Poly2D(
        np.array(
            [
                [10, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ]
        )
    )
