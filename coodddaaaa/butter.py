"""
Modified after sigy 1.5.3, M.L. 21/04/2023

Time / Fourier domain butterworth filter
warning : the fourier domain filter has a slightly different response that the time domain one
          comparing both reveals that the time domain filter may include a water-level that do not 
          not exist with fourier domain, this results in slight differences near the signal edges
          taper the waveform properly fixes the difference
TODO : use ba_analog ?
TODO : use scipy.fft => parallel
"""

from typing import Optional
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz
from scipy.fftpack import fftfreq, fft, ifft, fft2, ifft2
import numpy as np


class ButterworthFilter(object):
    """

    :param freqmin: lower frequency in Hz, or None for highpass filtering
    :param freqmax: upper frequency in Hz, or None for lowpass filtering
    :param sampling_rate: in Hz
    :param order: of the filter
    """

    _sos = None
    _sampling_rate = None

    def __init__(self, 
        freqmin: Optional[float], 
        freqmax: Optional[float],
        sampling_rate: Optional[float], 
        order: float = 4.):

        nyquist = 0.5 * sampling_rate
        self._freqmin = freqmin
        self._freqmax = freqmax
        self._order = order
        self._sampling_rate = sampling_rate

        if freqmin is None and freqmax is None:
            raise ValueError(freqmin, freqmax)

        elif freqmin is not None and freqmax is not None:
            self._sos = butter(order, [freqmin / nyquist, freqmax / nyquist],
                               output="sos", btype="band")

        elif freqmin is not None:
            self._sos = butter(order, [freqmin / nyquist],
                               output="sos", btype="high")

        elif freqmax is not None:
            self._sos = butter(order, [freqmax / nyquist],
                               output="sos", btype="low")

        else:
            raise ValueError(freqmin, freqmax)

    def timecall(self, data, zerophase=False, axis=-1):
        if not zerophase:
            filtered_data = sosfilt(sos=self._sos, x=data, axis=axis)
            # filtered_data = lfilter(b=self.b, a=self.a, x=data, axis=axis)

        else:
            filtered_data = sosfiltfilt(sos=self._sos, x=data, axis=axis)
    
    
        return filtered_data

    def response(self, npts, zerophase=False, input_domain="fft", qc=False):
        """
        Almost equivalent to timecall
        the response looks better than timecall, no water level applied
        """

        if input_domain == "fft":
            freqs = fftfreq(npts, 1. / self._sampling_rate)
            # equivalent to (except for freqs (0 to nyquist, no wrapping))
            # freqs, response = sosfreqz(self._sos, worN=npts, whole=True, fs=self._sampling_rate)

        elif input_domain == "rfft":
            raise Exception(
                'warning : the behavior of scipy.fftpack.rfft '
                'differs from scipy.fft.rfft')
            freqs = rfftfreq(npts, 1. / self._sampling_rate)

        else:
            raise NotImplementedError(input_domain)

        _, response = sosfreqz(self._sos, worN=freqs, whole=True, fs=self._sampling_rate)

        if zerophase:
            response = np.abs(response) ** 2.

        if qc:
            import matplotlib.pyplot as plt
            data = np.random.randn(npts)
            filtered_data = self.timecall(data=data, zerophase=zerophase)

            if input_domain == "fft":
                # freqs = fftfreq(npts, 1./self._sampling_rate)
                tfdata = fft(data)
                filtered_tfdata = fft(filtered_data)

            elif input_domain == "rfft":
                raise Exception(
                    'warning : the behavior of scipy.fftpack.rfft '
                    'differs from scipy.fft.rfft')
                # freqs = fftfreq(npts, 1. / self._sampling_rate)
                tfdata = rfft(data)
                filtered_tfdata = rfft(filtered_data)

            else:
                raise NotImplementedError(input_domain)

            expected_response = filtered_tfdata / tfdata

            plt.figure()
            plt.subplot(311, title=f"{zerophase}")
            plt.plot(freqs, expected_response.real, linewidth=3)
            plt.plot(freqs, response.real, linewidth=1)

            plt.subplot(312)
            plt.plot(freqs, expected_response.imag, linewidth=3)
            plt.plot(freqs, response.imag, linewidth=1)

            plt.subplot(313)
            plt.loglog(np.abs(freqs), np.abs(expected_response), linewidth=3)
            plt.loglog(np.abs(freqs), np.abs(response), linewidth=1)
            plt.show()

        return freqs, response

    def __call__(self, data, zerophase=False, axis=-1, input_domain="time"):
        """
        Returns the filtered data
        can be called on time domain (real or complex) data
        fft or rfft transformed data (use input_domain)

        :param data:
        :param zerophase:
        :param axis:
        :param input_domain:
        """
        if input_domain == "time":
            return self.timecall(data=data, zerophase=zerophase, axis=axis)

        elif input_domain in ["fft", "rfft"]:

            if input_domain == "rfft" and zerophase is False:
                raise Exception(
                    'warning : the behavior of scipy.fftpack.rfft '
                    'differs from scipy.fft.rfft')
                raise ValueError(f"{input_domain=}, {zerophase=} => complex => irfft not applicable")

            _, response = self.response(
                npts=len(data), zerophase=zerophase,
                input_domain=input_domain, qc=False)

            return data * response

        else:
            raise ValueError(input_domain)

    def show(self, fig, freqs=None, zerophase=False, **kwargs):
        freqs, response = sosfreqz(self._sos, worN=freqs, whole=False, fs=self._sampling_rate)

        ax = fig.add_subplot(121)
        bx = fig.add_subplot(122, sharex=ax)

        if zerophase:
            response = np.abs(response) ** 2.0

        ax.loglog(freqs, np.abs(response), **kwargs)
        bx.semilogx(freqs, np.angle(response), **kwargs)

        ax.set_ylabel('response modulus')
        bx.set_ylabel('response phase')

        for cx in [ax, bx]:
            ylim = cx.get_ylim()
            cx.plot(self._freqmin * np.ones(2), ylim, 'r--')
            cx.plot(self._freqmax * np.ones(2), ylim, 'r--')
            cx.grid(True, linestyle="--")
            cx.set_xlabel('frequency (Hz)')
        fig.suptitle(
            f'{self._freqmin},{self._freqmax},{self._order},{zerophase}')
        

class BandpassFilter(ButterworthFilter):
    """
    Shortcut for ButterworthFilter for band-pass filtering
    """
    def __init__(self, freqmin, freqmax, sampling_rate, order=4):
        ButterworthFilter.__init__(
            self, freqmin=freqmin, freqmax=freqmax,
            sampling_rate=sampling_rate, order=order)


class LowpassFilter(ButterworthFilter):
    """
    Shortcut for ButterworthFilter for low-pass filtering
    """

    def __init__(self, freqmax, sampling_rate, order=4):
        ButterworthFilter.__init__(
            self, freqmin=None, freqmax=freqmax,
            sampling_rate=sampling_rate, order=order)


class HighpassFilter(ButterworthFilter):
    """
    Shortcut for ButterworthFilter for high-pass filtering
    """
    def __init__(self, freqmin, sampling_rate, order=4):
        ButterworthFilter.__init__(
            self, freqmin=freqmin, freqmax=None,
            sampling_rate=sampling_rate, order=order)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.signal.windows import tukey

    npts = 1200
    sampling_rate = 1.0123456
    data = 1.0 * np.random.randn(npts)
    data *= tukey(len(data), 0.2)

    freqmin = 0.03
    freqmax = 0.08
    fftfreqs = fftfreq(npts, 1. / sampling_rate)

    bp = BandpassFilter(freqmin=freqmin, freqmax=freqmax, sampling_rate=sampling_rate, order=4)

    bp.show(plt.figure(), zerophase=False)
    bp.show(plt.figure(), zerophase=True)

    ax = plt.gcf().axes[0]
    ax.plot([freqmin, freqmin], ax.get_ylim(), 'r--')
    ax.plot([freqmax, freqmax], ax.get_ylim(), 'r--')

    plt.figure()
    plt.plot(data, 'k')
    plt.plot(bp(data, zerophase=True), "b", linewidth=3)
    plt.plot(ifft(bp(fft(data), zerophase=True, input_domain="fft")).real, "g-")
    # plt.plot(irfft(bp(rfft(data), zerophase=True, input_domain="rfft")).real, "m--")

    plt.show()
