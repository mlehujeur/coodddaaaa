"""
Copyright (c) 2023 maximilien.lehujeur

Finds the maximum of a random array with subsample precision
by looking for the zero crossing of the
first order finite difference derivative
"""
from typing import Optional
import numpy as np
from matplotlib.axes import Axes


def hypermax(time_array: np.ndarray, function_array: np.ndarray, axqc: Optional[Axes] = None, assume_t_growing: bool = False):
    """
    :param time_array: time array
    :param function_array: function array
    :param axqc: matplotlib ax or None for visual qc
    :param assume_t_growing:
    """
    if not assume_t_growing:
        assert (time_array[1:] > time_array[:-1]).all(), "t must be strictly growing"

    imax = np.argmax(function_array)
    tmax, fmax = time_array[imax], function_array[imax]
    
    if imax == 0:
        return time_array[0]

    elif imax == len(time_array) - 1:
        return time_array[-1]

    tt = (0.5 * (time_array[imax: imax + 2] + time_array[imax - 1: imax + 1]))
    ff = (function_array[imax: imax + 2] - function_array[imax - 1: imax + 1]) / (time_array[imax: imax + 2] - time_array[imax - 1: imax + 1])

    ip = np.argsort(ff)
    thypermax = np.interp(0., xp=ff[ip], fp=tt[ip])

    if axqc is not None:
        axqc.plot(time_array, function_array, "k+-", alpha = 0.4)
        axqc.plot(tmax, fmax, 'ko')
        # axqc.plot(dt, df, "r+-", alpha = 0.4)
        axqc.plot(tt, ff, "r", alpha = 1.0)
        axqc.plot(thypermax, 0, "r*", alpha = 1.0)
        axqc.plot(thypermax * np.ones(2), axqc.get_ylim(), 'r')
        axqc.grid(True)

    return thypermax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = np.unique(np.random.randn(50))
    f = np.sinc(t)

    tmax = hypermax(t, f, axqc=plt.gca())

    print('true max : ', 0.)
    print('max : ', t[np.argmax(f)])
    print('hypermax : ',  tmax)
    plt.show()
