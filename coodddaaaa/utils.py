"""
Copyright (c) 2023 maximilien.lehujeur
"""


import time
import numpy as np


class Timer:
    """Counts the execution time under the "with" statement

    :param message:
    """

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        end = time.perf_counter()
        print(f'Timer[{self.message}]: {(end - self.start) * 1000.:.2f} ms')


def polyspace(xmin: float, xmax: float, nx: int, pwr: float):
    """A power-law to refine resolution of a stretching grid search near zero

    :param xmin: min value
    :param xmax: max value
    :param nx: number of points
    :param pwr: power coefficient, increase pwr to refine the resolution near 0

    """
    assert pwr > 0, ValueError(pwr)

    tmin = np.sign(xmin) * np.abs(xmin) ** (1. / pwr)
    tmax = np.sign(xmax) * np.abs(xmax) ** (1. / pwr)
    t = np.linspace(tmin, tmax, nx)
    return np.sign(t) * np.abs(t) ** pwr


class TukeyWindow:
    """
    A parameterizable 4 points Tukey function
    :param t0 ... t3: times of the corners of the Tukey window
    """
    def __init__(self, t0:float, t1:float, t2:float, t3:float):
        """

        """
        assert t0 <= t1 <= t2 <= t3
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

    def _growing(self, t):
        return (1. - np.cos(np.pi * (t - self.t0) / (self.t1 - self.t0))) / 2.0

    def _decreasing(self, t):
        return (1. + np.cos(np.pi * (t - self.t2) / (self.t3 - self.t2))) / 2.0

    def __call__(self, t: np.ndarray):
        """
        Evaluate the taper function at t

        :param t: does not need to be sorted or regularly spaced
        """
        i = np.argsort(t)

        j0, j1, j2, j3 = \
            np.searchsorted(
                a=t[i], v=[self.t0, self.t1, self.t2, self.t3])

        y = np.zeros_like(t)
        y[i[:j0]] = 0.
        y[i[j0:j1]] = self._growing(t[i[j0:j1]])
        y[i[j1:j2]] = 1.
        y[i[j2:j3]] = self._decreasing(t[i[j2:j3]])
        y[i[j3:]] = 0.
        return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.random.randn(10000)
    with Timer('TukeyWindow'):
        taper = TukeyWindow(-0.5, -0.25, 0.25, 0.33)
        y = taper(x)

    plt.figure()
    plt.plot(x, y, '.')

    with Timer('Polyspace'):
        x = polyspace(-0.01, 0.03, 100, 2)
    plt.figure()
    plt.plot(x)
    plt.show()
