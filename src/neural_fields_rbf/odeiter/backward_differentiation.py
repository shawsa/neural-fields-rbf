from abc import abstractproperty
from collections import deque
from itertools import islice
from scipy.optimize import root

from .single_step import ImplicitEuler
from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator


def default_root_finder(fun, x0):
    sol = root(fun=fun, x0=x0)
    if not sol.success:
        raise ValueError(sol)
    return sol.x


class BackwardDifferentiationAbstract(TimeIntegrator):
    def __init__(
        self,
        seed: TimeIntegrator,
        seed_steps_per_step: int,
        root_finder=default_root_finder,
    ):
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 1
        self.root_finder = root_finder

    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def order(self) -> int:
        ...

    @abstractproperty
    def us_coeffs(self) -> list[float]:
        ...

    @abstractproperty
    def f_coeff(self) -> float:
        ...

    def update(self, t, us, rhs, delta_t):
        const_vec = sum(c * u for c, u in zip(self.us_coeffs[::-1], us))

        def func(x):
            return -x + const_vec + delta_t * self.f_coeff * rhs(t + delta_t, x)

        return self.root_finder(func, us[-1])

    def solution_generator(self, u0, rhs, time: TimeDomain):
        seed_steps = self.order - 1
        seed_time = TimeRay(
            time.start,
            time.spacing / self.seed_steps_per_step,
        )
        t = time.start
        us = deque(maxlen=self.order)
        for u in islice(
            self.seed.solution_generator(u0, rhs, seed_time),
            0,
            self.seed_steps_per_step * seed_steps + 1,
            self.seed_steps_per_step,
        ):
            yield u
            us.append(u)

        for t in time.array[seed_steps:-1]:
            u = self.update(t, us, rhs, time.spacing)
            yield u
            us.append(u)


BDF1 = ImplicitEuler


class BDF2(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF2 (seed: {self.seed.name})"

    @property
    def order(self):
        return 2

    @property
    def us_coeffs(self) -> list[float]:
        return [4 / 3, -1 / 3]

    @property
    def f_coeff(self) -> float:
        return 2 / 3


class BDF3(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF3 (seed: {self.seed.name})"

    @property
    def order(self):
        return 3

    @property
    def us_coeffs(self) -> list[float]:
        return [18 / 11, -9 / 11, 2 / 11]

    @property
    def f_coeff(self) -> float:
        return 6 / 11


class BDF4(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF4 (seed: {self.seed.name})"

    @property
    def order(self):
        return 4

    @property
    def us_coeffs(self) -> list[float]:
        return [48 / 25, -36 / 25, 16 / 25, -3 / 25]

    @property
    def f_coeff(self) -> float:
        return 12 / 25


class BDF5(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF5 (seed: {self.seed.name})"

    @property
    def order(self):
        return 5

    @property
    def us_coeffs(self) -> list[float]:
        return [300 / 137, -300 / 137, 200 / 137, -75 / 137, 12 / 137]

    @property
    def f_coeff(self) -> float:
        return 60 / 137


class BDF6(BackwardDifferentiationAbstract):
    @property
    def name(self):
        return f"BDF6 (seed: {self.seed.name})"

    @property
    def order(self):
        return 6

    @property
    def us_coeffs(self) -> list[float]:
        return [360 / 147, -450 / 147, 400 / 147, -225 / 147, 72 / 147, -10 / 147]

    @property
    def f_coeff(self) -> float:
        return 60 / 147
