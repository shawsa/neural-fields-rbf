from .single_step import Trapezoidal
from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator
from abc import abstractproperty
from collections import deque
from itertools import islice
from scipy.optimize import root


def default_root_finder(fun, x0):
    sol = root(fun=fun, x0=x0)
    if not sol.success:
        raise ValueError(sol)
    return sol.x


class AdamsMoultonAbstract(TimeIntegrator):
    def __init__(
        self,
        seed: TimeIntegrator,
        seed_steps_per_step: int,
        root_finder=default_root_finder,
    ):
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 2
        self.root_finder = root_finder

    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def fs_coeffs(self) -> list[float]:
        ...

    def update(self, t, u, rhs, fs, delta_t):
        const_vec = u + delta_t * sum(
            c * f for c, f in zip(self.fs_coeffs[-1:0:-1], fs)
        )

        def func(x):
            return -x + const_vec + delta_t * self.fs_coeffs[0] * rhs(t + delta_t, x)

        return self.root_finder(func, u)

    def solution_generator(self, u0, rhs, time: TimeDomain):
        seed_steps = self.order - 1
        seed_time = TimeRay(
            time.start,
            time.spacing / self.seed_steps_per_step,
        )
        t = time.start
        fs = deque(maxlen=self.order - 1)
        for u in islice(
            self.seed.solution_generator(u0, rhs, seed_time),
            0,
            self.seed_steps_per_step * seed_steps + 1,
            self.seed_steps_per_step,
        ):
            yield u
            fs.append(rhs(t, u))
            t += time.spacing

        for t in time.array[seed_steps:-1]:
            u = self.update(t, u, rhs, fs, time.spacing)
            yield u
            fs.append(rhs(t + time.spacing, u))


AM1 = Trapezoidal


class AM2(AdamsMoultonAbstract):
    @property
    def order(self):
        return 3

    @property
    def name(self):
        return f"AM2 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [5 / 12, 8 / 12, -1 / 12]


class AM3(AdamsMoultonAbstract):
    @property
    def order(self):
        return 4

    @property
    def name(self):
        return f"AM3 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [9 / 24, 19 / 24, -5 / 24, 1 / 24]


class AM4(AdamsMoultonAbstract):
    @property
    def order(self):
        return 5

    @property
    def name(self):
        return f"AM4 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]
