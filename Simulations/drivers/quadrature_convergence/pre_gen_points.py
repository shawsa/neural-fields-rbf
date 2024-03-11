"""
Run convergence Tests for number of points
"""
from dataclasses import dataclass
from math import ceil
import numpy as np
from os import listdir
import pickle
from rbf.points.unit_square import hex_limit_density, UnitSquare
from rbf.quadrature import LocalQuad
from rbf.rbf import RBF, PHS
import time
from tqdm import tqdm
from utils import (
    hex_stencil_min,
    poly_stencil_min,
)


PATH = "data/pregen/"
STENCIL_SIZE_FACTOR = 2


def make_file_name(poly_deg: int, n: int):
    return PATH + f"d{poly_deg}_n{n}_timestamp{round(time.time())}.pickle"


@dataclass
class CachedQF:
    rbf: RBF
    poly_deg: int
    stencil_size: int
    points: np.ndarray[float]
    weights: np.ndarray[float]


def get_stencil_size(poly_deg: int) -> int:
    return hex_stencil_min(ceil(STENCIL_SIZE_FACTOR * poly_stencil_min(poly_deg)))


def pregens():
    for filename in listdir(PATH):
        with open(PATH + filename, "rb") as f:
            cached_qf = pickle.load(f)
        yield cached_qf


if __name__ == "__main__":
    rbf = PHS(3)
    poly_degs = range(1, 7)
    h_targets = np.logspace(-6, -8, 11, base=2)
    repeats = 5

    for _ in tqdm(range(repeats), leave=True, position=0, unit="trial"):
        for h_target in (hs_prog := tqdm(h_targets[::-1], leave=False, position=1)):
            n = hex_limit_density(h_target)
            hs_prog.set_description(f"{n=}")
            us = UnitSquare(
                n,
                verbose=True,
                tqdm_kwargs={
                    "position": 2,
                    "leave": False,
                },
            )
            for poly_deg in (
                deg_prog := tqdm(poly_degs[::-1], leave=False, position=2)
            ):
                stencil_size = get_stencil_size(poly_deg)
                deg_prog.set_description(f"{poly_deg=}, k={stencil_size}")
                qf = LocalQuad(
                    us.points,
                    rbf,
                    poly_deg,
                    stencil_size,
                    verbose=True,
                    tqdm_kwargs={
                        "position": 3,
                        "leave": False,
                    },
                )

                cached_qf = CachedQF(
                    rbf=rbf,
                    poly_deg=poly_deg,
                    stencil_size=stencil_size,
                    points=us.points,
                    weights=qf.weights,
                )
                file_name = make_file_name(poly_deg=poly_deg, n=len(us.points))
                with open(file_name, "wb") as f:
                    pickle.dump(cached_qf, f)
