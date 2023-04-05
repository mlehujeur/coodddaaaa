"""
Copyright (c) 2023 maximilien.lehujeur
"""


import time
import numpy as np


class Timer:
    """Counts the execution time under the "with" statement"""

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        end = time.perf_counter()
        print(f'Timer[{self.message}]: {(end - self.start) * 1000.:.2f} ms')


def polyspace(xmin: float, xmax: float, nx: int, pwr: float):
    """a power low to refine resolution near zero"""
    u = np.linspace(-1., 1., nx)
    u = np.sign(u) * np.abs(u) ** pwr
    return 0.5 * (u + 1.) * (xmax - xmin) + xmin
