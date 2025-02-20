import sympy as sym

import numpy as np
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus

from rbf.surface import TriMesh


# R, r = sym.symbols("R r", real=True)
R, r = 3, 1
p, th = sym.symbols("\\phi \\theta", real=True)

x = (R + r * sym.cos(th)) * sym.cos(p)
y = (R + r * sym.cos(th)) * sym.sin(p)
z = r * sym.sin(th)

J = r * sym.Abs(R + r * sym.cos(th))

p2, th2 = sym.symbols("\\phi_2 \\theta_2", real=True)
dist_phi = sym.sin((p - p2) / 2)
dist_theta = sym.sin((th - th2) / 2)
dist_squared = dist_phi**2 + dist_theta**2
w = 1 - dist_squared

t = sym.symbols("t", real=True)
fu = (sym.sin(7 * (p2 + t)) * sym.sin(th2) + sym.Rational(11, 10)) / sym.Rational(
    23, 10
)

int_op = sym.integrate(
    sym.integrate(w * fu * J, (th2, -sym.pi, sym.pi)), (p2, -sym.pi, sym.pi)
).simplify()


z = sym.symbols("z", real=True)
gamma = 10
threshold = sym.Rational(2, 10)
f = 1 / (1 + sym.exp(-(z - threshold) * gamma))
f_inv = threshold - sym.log(1 / z - 1) / gamma

u = f_inv.subs(z, fu.subs(p2, p).subs(th2, th))
ut = u.diff(t).simplify()

F = ut + u - int_op


exact = sym.lambdify((t, p, th), u)
forcing = sym.lambdify((t, p, th), F)
firing_rate = sym.lambdify(z, f)
kernel = sym.lambdify((p, th, p2, th2), w)


def cart_to_param(points):
    if points.ndim == 1:
        x, y, z = points
    else:
        x, y, z = points.T
    phis = np.arctan2(y, x)
    thetas = np.arctan2(z, np.sqrt(x**2 + y**2) - R)
    return phis, thetas


def dist(points, points0):
    phis, thetas = cart_to_param(points)
    phi0, theta0 = cart_to_param(points[0])

    return np.sin(.5*(phis - phi0))**2 + np.sin(.5*(thetas - theta0))**2


def weight_kernel(r):
    return 1 - r


if __name__ == "__main__":
    N = 10_000
    torus = SpiralTorus(N, R=R, r=r)
    N = torus.N
    points = torus.points
    vor = LocalSurfaceVoronoi(
        torus.points,
        torus.normals,
        torus.implicit_surf,
    )
    trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
    triangles = pv.PolyData(points, [(3, *f) for f in trimesh.simplices])

    phis = np.arctan2(points[:, 1], points[:, 0])
    thetas = np.arctan2(points[:, 2], np.sqrt(points[:, 0]**2 + points[:, 1]**2) - R)

    plotter = pv.Plotter()
    plotter.add_mesh(
        triangles,
        show_edges=False,
        show_vertices=False,
        scalars=thetas,
        show_scalar_bar=True,
    )
    plotter.show()

    plotter = pv.Plotter()
    plotter.add_mesh(
        triangles,
        show_edges=False,
        show_vertices=False,
        scalars=phis,
        show_scalar_bar=True,
    )
    plotter.show()

    for p, t in [(0, 0), (1.1, 0), (0, 1.1), (-3, 2)]:
        plotter = pv.Plotter()
        plotter.add_mesh(
            triangles,
            show_edges=False,
            show_vertices=False,
            scalars=kernel(phis, thetas, p, t),
            show_scalar_bar=True,
        )
        plotter.show()

    for t in [0, 1, 2, 3]:
        plotter = pv.Plotter()
        plotter.add_mesh(
            triangles,
            show_edges=False,
            show_vertices=False,
            scalars=exact(t, phis, thetas),
            show_scalar_bar=True,
        )
        plotter.show()
