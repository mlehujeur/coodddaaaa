"""
Copyright (c) 2023 maximilien.lehujeur

Fourier domain oversampling, for the sake of simplicity, 
this program can increase the number of samples only by 2**n, where n is an integer.

The oversampling is performed in the Fourier domain, 
- for optimal use, it is preferred to use the fft_oversamp
  if the signal is already in Fourier domain
- make sure the signal is properly detrended and tapered at its edges prior to FFT
  otherwise, you might observe wiggles at the edges of the signal after oversampling.
  (remember that fft assume a periodization of the signal in time)

"""

import numpy as np
from scipy.fftpack import fft, ifft


def oversamp(t0: float, dt: float, data: np.ndarray,
             npow2: int, axis: int = -1, demean: bool = False) \
        -> (np.ndarray, np.ndarray):
    """
    time domain version of fft_oversamp
    :param t0: start time, sec
    :param dt: sampling interval, sec
    :param data: time domain data array (1d or more)
    :param npow2: oversamp by 2 ** npow2
    :param axis: axis along which to oversample the signal
    :return to: the new time vector
    :return datao: the oversample data
    """
    nt = data.shape[axis]
    t_over = t0 + np.arange(nt * 2 ** npow2) * (dt / (2 ** npow2))
    if demean:
        m = data.mean(axis=axis)
    else:
        m = 0.

    # factor = exp(npow2 * log(2))
    # npow2 = log(factor) / log(2)
    data_over = ifft(
        fft_oversamp(
            fft(data - m, axis=axis),
            npow2=npow2,
            axis=axis),
        axis=axis).real + m
    return t_over, data_over


def fft_oversamp(fft_data: np.ndarray, npow2: int = 1, axis: int = -1) -> np.ndarray:
    """
    oversamp a signal by padding it with zeros in the FFT domain
    the number of sample is multiplied by 2 ** npow2 (default 2**1)

    :param fft_data: output of fft
    :param npow2: oversampling rate expressed as a power of 2
    :param axis: the axis along which to perform oversampling
    :return: oversample data in fft domain
    """

    n = 2 ** npow2

    npts = fft_data.shape[axis]

    i_first_negative_freq = npts // 2 + npts % 2
    n_positive_freqs = i_first_negative_freq
    n_negative_freqs = npts - n_positive_freqs

    new_shape = list(fft_data.shape)
    new_shape[axis] = n * npts
    new_fft_data = np.zeros(new_shape, complex)

    view = fft_data.swapaxes(0, axis)
    new_view = new_fft_data.swapaxes(0, axis)

    new_view[:n_positive_freqs, ...] = n * view[:i_first_negative_freq, ...]
    new_view[-n_negative_freqs:, ...] = n * view[i_first_negative_freq:, ...]

    return new_fft_data


if __name__ == '__main__':

    from scipy.signal import butter, sosfiltfilt
    from scipy.signal.windows import tukey
    import matplotlib.pyplot as plt

    nt = 127  # number of samples
    dt = 0.1  # sampling interval in sec
    t0 = -10.012351503  # starttime in sec

    t = t0 + np.arange(nt) * dt
    # ny = 0.5 / dt  # nyquist, Hz

    # prepare bandpass filter and time taper
    sos = butter(4.0,  # order
            [0.1, 0.5],  # fmin, fmax relative to nyquist
            output="sos", btype="band")
    taper = tukey(nt, 0.1)

    # generate random signal
    y = np.random.randn(nt)

    # bandpass / taper
    y = sosfiltfilt(sos=sos, x=y)
    y *= taper  

    # oversamp
    t1, y1 = oversamp(t0=t0, dt=dt, data=y, npow2=1, axis=-1, demean=True)

    plt.figure()
    plt.plot(t, y, 'ko-')
    plt.plot(t1, y1, 'r')
    plt.show()
