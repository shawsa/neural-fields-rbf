from .time_domain import TimeDomain, TimeRay
from .time_integrator import TimeIntegrator
from abc import abstractproperty
from collections import deque
from itertools import islice


class AdamsBashforthAbstract(TimeIntegrator):
    def __init__(self, seed: TimeIntegrator, seed_steps_per_step):
        self.seed = seed
        self.seed_steps_per_step = seed_steps_per_step
        self.seed_steps = self.order - 1

    @abstractproperty
    def name(self) -> str:
        ...

    @abstractproperty
    def fs_coeffs(self) -> list[float]:
        ...

    def update(self, u, fs, delta_t):
        return u + delta_t * sum(c * f for c, f in zip(self.fs_coeffs[::-1], fs))

    def solution_generator(self, u0, rhs, time: TimeDomain):
        seed_steps = self.order - 1
        seed_time = TimeRay(
            time.start,
            time.spacing / self.seed_steps_per_step,
        )
        t = time.start
        fs = deque(maxlen=self.order)
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
            u = self.update(u, fs, time.spacing)
            yield u
            fs.append(rhs(t + time.spacing, u))


class AB2(AdamsBashforthAbstract):
    @property
    def order(self):
        return 2

    @property
    def name(self):
        return f"AB2 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [3 / 2, -1 / 2]


class AB3(AdamsBashforthAbstract):
    @property
    def order(self):
        return 3

    @property
    def name(self):
        return f"AB3 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [23 / 12, -16 / 12, 5 / 12]


class AB4(AdamsBashforthAbstract):
    @property
    def order(self):
        return 4

    @property
    def name(self):
        return f"AB4 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [55 / 24, -59 / 24, 37 / 24, -9 / 24]


class AB5(AdamsBashforthAbstract):
    @property
    def order(self):
        return 5

    @property
    def name(self):
        return f"AB5 (seed: {self.seed.name})"

    @property
    def fs_coeffs(self) -> list[float]:
        return [1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720]
