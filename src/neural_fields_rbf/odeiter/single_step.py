from abc import abstractmethod
from scipy.optimize import root

from .time_domain import TimeDomain
from .time_integrator import TimeIntegrator


class SingleStepMethod(TimeIntegrator):
    @abstractmethod
    def update(self, t, u, f, h):
        raise NotImplementedError

    def solution_generator(self, u0, rhs, time: TimeDomain):
        u = u0
        yield u
        for t in time.array[:-1]:
            u = self.update(t, u, rhs, time.spacing)
            yield u


class Euler(SingleStepMethod):
    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Euler"

    def update(self, t, u, f, h):
        return u + h * f(t, u)


class EulerDelta(SingleStepMethod):
    def __init__(self, delta_time, delta_profile):
        self.delta_time = delta_time
        self.delta_profile = delta_profile

    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Euler-Delta"

    def update(self, t, u, f, h):
        u_new = u + h * f(t, u)
        if abs(t - self.delta_time) < h / 2:
            u_new += self.delta_profile
        return u_new


class RK4(SingleStepMethod):
    @property
    def name(self):
        return "RK4"

    @property
    def order(self):
        return 4

    def update(self, t, u, f, h):
        k1 = f(t, u)
        k2 = f(t + h / 2, u + h / 2 * k1)
        k3 = f(t + h / 2, u + h / 2 * k2)
        k4 = f(t + h, u + h * k3)
        return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# implicit methods
def default_root_finder(fun, x0):
    sol = root(fun=fun, x0=x0)
    if not sol.success:
        raise ValueError(sol)
    return sol.x


class ImplicitEuler(SingleStepMethod):
    def __init__(self, root_finder=default_root_finder):
        self.root_finder = root_finder

    @property
    def order(self):
        return 1

    @property
    def name(self):
        return "Implicit Euler"

    def update(self, t, u, f, h):
        def func(x):
            return -x + u + h * f(t + h, x)

        return self.root_finder(func, u)


class Trapezoidal(SingleStepMethod):
    def __init__(self, root_finder=default_root_finder):
        self.root_finder = root_finder

    @property
    def order(self):
        return 2

    @property
    def name(self):
        return "Trapezoidal"

    def update(self, t, u, f, h):
        fn = f(t, u)

        def func(x):
            return -x + u + h / 2 * (fn + f(t + h, x))

        return self.root_finder(func, u)
