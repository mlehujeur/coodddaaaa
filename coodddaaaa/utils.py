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
    assert pwr > 0, ValueError(pwr)

    tmin = np.sign(xmin) * np.abs(xmin) ** (1. / pwr)
    tmax = np.sign(xmax) * np.abs(xmax) ** (1. / pwr)
    t = np.linspace(tmin, tmax, nx)
    return np.sign(t) * np.abs(t) ** pwr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    with Timer('test'):
        x = polyspace(-0.01, 0.03, 100, 2)

    plt.plot(x)
    plt.show()
