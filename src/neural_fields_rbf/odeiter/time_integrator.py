"""A time-integration class for solving ODEs numerically."""
from abc import ABC, abstractmethod, abstractproperty
from tqdm import tqdm
from .time_domain import TimeDomain


class TimeIntegrator(ABC):
    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    @abstractproperty
    def order(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def solution_generator(self, u0, rhs, time: TimeDomain):
        raise NotImplementedError

    def solve(self, u0, rhs, time: TimeDomain):
        return list(self.solution_generator(u0, rhs, time))

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in self.solution_generator(u0, rhs, time):
            pass
        return u


class TqdmWrapper:
    def __init__(self, solver: TimeIntegrator):
        self.solver = solver

    def solve(self, u0, rhs, time: TimeDomain):
        return list(
            tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps)
        )

    def t_final(self, u0, rhs, time: TimeDomain):
        for u in tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps):
            pass
        return u

    def solution_generator(self, u0, rhs, time: TimeDomain):
        yield from tqdm(self.solver.solution_generator(u0, rhs, time), total=time.steps)


