"""A time-discretization for an ODE solver."""
import math
import numpy as np


class TimeDomain:
    def __init__(self, start, spacing, steps):
        self.start = start
        self.spacing = spacing
        self.steps = steps
        self.initialze_array()

    def initialze_array(self):
        self.array = self.start + self.spacing * np.arange(self.steps + 1)

    def __iter__(self):
        yield from self.array


class TimeDomain_Start_Stop_MaxSpacing(TimeDomain):
    def __init__(self, start, stop, max_spacing):
        self.start = start
        self.steps = math.ceil((stop - start) / max_spacing)
        self.spacing = (stop - start) / self.steps
        self.initialze_array()


class TimeDomain_Start_Stop_Steps(TimeDomain):
    def __init__(self, start, stop, steps):
        self.start = start
        self.steps = steps
        self.spacing = (stop - start) / steps
        self.initialze_array()


class Ray:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.start + key * self.step
        if not isinstance(key, slice):
            raise ValueError
        start = key.start or 0
        step = key.step or 1
        return Ray(self[start], self.step * step)

    def __iter__(self):
        index = 0
        while True:
            yield self.start + index * self.step
            index += 1

    def __repr__(self):
        return f"Ray(start={self.start}, step={self.step})"


class TimeRay(TimeDomain):
    def __init__(self, start, spacing):
        self.start = start
        self.spacing = spacing

    @property
    def array(self):
        return Ray(self.start, self.spacing)

    def __iter__(self):
        return self.array.__iter__()
