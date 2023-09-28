"""
Copyright (c) 2023 maximilien.lehujeur
"""

from typing import Union, Optional, Literal
import numpy as np
from coodddaaaa.interp1d import LinearInterpolator1d, CubicInterpolator1d, RFFTInterpolator1d
from coodddaaaa.hypermax import hypermax
from scipy.sparse import block_diag


class Stretcher:
    r"""
    An object to compute stretched signals and to perform stretching correlation as defined in Weaver et al., 2011.

    .. math:: X(\varepsilon) =
        \frac
            {\int{y^{ref}(t \times (1 + \varepsilon)) \cdot y(t) dt}}
            {\sqrt{
                \int{y^{ref}(t) dt}
                \int{y(t) dt}
                }}


    The object pre-computes the interpolation operator.
    The user can pre-compute and store the stretched basis of the reference signal.
    This basis can then be provided for stretching correlation with a new signal.
    This object can also compute the stretching between all pairs of signals in a b-scan.

    :param t0: time of first sample
    :param dt: sampling interval
    :param nt: number of samples
    :param eps: epsilon array
    :param norm: use it to compute normalized correlation.
                 warning : for stretching only, use norm = False
    :param interp_kind: which interpolator to use for stretching, among 'linear', "cubic", 'fourier'
    """
    def __init__(self,
                 t0: float, dt: float, nt: int,
                 eps: np.ndarray,
                 norm: bool = False,
                 interp_kind: Literal['linear', "cubic", 'fourier'] = "cubic"):
        """
        Initiate the stretcher and the interpolator on a fixed interpolation grid.

        """
        assert interp_kind in ['linear', 'cubic', 'fourier']

        self.t0, self.nt, self.dt = t0, nt, dt
        self.eps = eps   # 1d, shape (len(eps), )
        self.norm = norm
        self.interp_kind = interp_kind

        # Time array = nodes at which the function to stretch is defined
        self.t = t0 + np.arange(nt) * dt   # 1d, shape (nt, )

        # Compute the stretching time grid for all values in eps
        # = points at which to evaluate the function for stretching
        self.stretch_time = self.t * (1. + self.eps[:, np.newaxis])  # 2d array, shape (len(eps), nt)

        # compute the interpolation operator once for all (based on scipy.sparse matrices)
        if interp_kind == "linear":
            self.interpolator = LinearInterpolator1d(
                x0=t0, nx=nt, dx=dt,  # nodes
                xi=self.stretch_time.flat[:],  # interpolation points
                )

        elif interp_kind == "cubic":
            t0 = self.t[0]
            nt = len(self.t)
            dt = self.t[1] - self.t[0]
            assert ((self.t - (np.arange(nt) * dt + t0)) / dt <= 1e-6) .all()

            self.interpolator = CubicInterpolator1d(
                x0=t0, nx=nt, dx=dt,  # nodes
                xi=self.stretch_time.flat[:],  # interpolation points
                )

        elif interp_kind == "fourier":
            t0 = self.t[0]
            nt = len(self.t)
            dt = self.t[1] - self.t[0]
            assert ((self.t - (np.arange(nt) * dt + t0)) / dt <= 1e-6) .all()

            self.interpolator = RFFTInterpolator1d(
                x0=t0, nx=nt, dx=dt,  # nodes
                xi=self.stretch_time.flat[:],  # interpolation points
                )

        else:
            raise ValueError(interp_kind)

    def stretch(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the stretched basis functions from a signal x

        :param x: the input signal (reference), np.ndarray, 1d, shape (nt, )
        :return x_stretched: the stretched version of x for all values in self.eps, np.ndarray 2d, shape (neps, nt)
        """

        x_stretched = np.zeros_like(self.stretch_time)
        x_stretched.flat[:] = self.interpolator(x)

        if self.norm:
            # normalize the stretched function now for efficiency
            # instead of doing it in the correlation
            # => WARNING : this will affect the amplitudes of the stretched data
            #              for stretching only, user must use norm=False

            # norm = np.linalg.norm(x_stretched, axis=1)
            norm = (x_stretched ** 2).sum(axis=1) ** -0.5

            # NB : I've tried Numba (this ref) => no gain at all
            # https://stackoverflow.com/questions/30437947/
            # most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray

            x_stretched *= norm[:, np.newaxis]

        return x_stretched

    def corr(self, x: np.ndarray, x_stretched: np.ndarray) -> np.ndarray:
        """
        Stretching correlation of x with a basis of stretched versions of the reference signal

        :param x: signal(s) to be correlated to the reference, np.ndarray,
            either one single signal, 1d, shape (nt, )
            or a bscan, 2d, shape (ntraces, nt)
        :param x_stretched: stretched reference from self.stretch, np.ndarray, 2d, shape (neps, nt, )
        :return c: correlation function np.ndarray,
            either 1d, shape (neps, ) if x is 1d
            or 2d, shape (neps, ntraces) if x is 2d
        """
        if x.ndim == 1:
            assert x.shape == (self.nt, ), \
                f'Shape Error : x must be 1 trace of shape (nt={self.nt}, )'

        elif x.ndim == 2:
            assert x.shape[1] == self.nt, \
                f'Shape Error : x must be a bscan of shape (ntraces, nt={self.nt})'

        else:
            raise ValueError('Shape Error, x must be 1d (for single signal) or 2d (for a bscan)')

        c = x_stretched.dot(x.T) * self.dt  # np.ndarray, shape (neps, ntraces)

        if self.norm:
            # the normalization relative to x_stretched
            # is already done
            if x.ndim == 1:
                # faster?
                c /= x.dot(x) ** 0.5 * self.dt
            elif x.ndim == 2:
                c /= (x * x).sum(axis=-1) ** 0.5 * self.dt  # shape (ntraces, )
            else:
                raise Exception('programming error')
        return c

    def corrmax(self, c: np.ndarray) -> (Union[float, np.ndarray], Union[float, np.ndarray]):
        """
        Find the maximum of the correlation function with subsample precision

        :param c: correlation function(s) from self.corr
            1d for a single signal, shape (neps, )
            2d for a bscan, shape (neps, ntraces)
        :return emax: best epsilon value, dimensionless, it corresponds to dt/t
            float if c is 1d
            1d array, shape (ntraces, ) if c is 2d
        :return cmax: max correlation, dimensionless, normalized if norm was True in __init__
            float if c is 1d
            1d array, shape (ntraces, ) if c is 2d
        """
        if c.ndim == 1:
            epsmax = hypermax(
                time_array=self.eps,
                function_array=c,
                assume_t_growing=True)
            cmax = c.max()

        elif c.ndim == 2:
            neps, ntraces = c.shape
            assert neps == len(self.eps), f"Shape Error, c must be of shape (neps={len(self.eps)}, ntraces)"

            epsmax = np.zeros(ntraces, float)
            for i in range(ntraces):
                # TODO : implement 2d version of hypermax to avoid the loop ?
                epsmax[i] = hypermax(
                    time_array=self.eps,
                    function_array=c[:, i],
                    assume_t_growing=True)
            cmax = c.max(axis=0)  # one max per trace

        else:
            raise ValueError(
                f'Shape Error, c must be 1d (single trace) or 2d (bscan)')

        return epsmax, cmax

    def corr_all_with_all(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Correlate all possible pairs of signals in a bscan

        :param data: the bscan, one trace per row, same sampling (=self.t), 2d, shape (ntraces, nt)
        :return c_triu: the max correlation coefficients for all pairs (upper triangle only)
        :return e_triu: the best stretching coefficients for all pairs (upper triangle only)
        use self.triu2dence to get the full matrices

        c = Stretcher.triu2dense(c_triu, symetric=True, diag=1.0)
        e = Stretcher.triu2dense(e_triu, symetric=False, diag=0.0)
        """

        ntraces, nsamps = data.shape

        c_triu = np.zeros(ntraces * (ntraces - 1) // 2)
        e_triu = np.zeros(ntraces * (ntraces - 1) // 2)
        # itriu, jtriu = np.triu_indices(ntraces, 1)

        n = 0
        for i in range(ntraces - 1):
            # print(f'{i+1}/{ntraces - 1}')
            # stretch new reference
            y = data[i, :]
            y_stretched = self.stretch(y)

            # correlate all remaining traces to this new reference
            m = ntraces - i - 1
            cijs = self.corr(x=data[i + 1:, :], x_stretched=y_stretched)
            e_triu[n: n+m], c_triu[n: n+m] = self.corrmax(cijs)
            n += m

        return c_triu, e_triu

    @staticmethod
    def triu2dense(x_triu: np.ndarray, symetric: bool, diag: float) -> np.ndarray:
        """
        Convert upper triangle matrix to square matrix

        :param x_triu: a flat upper triangle without diagonal, 1d, np.ndarray, shape (ntraces * (ntraces - 1) / 2, )
        :param symetric: to impose symetry (True) or anti-symetry (False)
        :param diag: the value to put on the diagonal
        :return x:
            a square matrix with x_triu on its upper triangle, shape (ntraces, ntraces)
            diag on its diagonal
            +-x_triu on its lower triangle
        """
        # 2n = (x * (x -1))
        # 2n = x ** 2 - x
        # x ** 2 - x - 2n = 0
        # d = 2 + 8 * n
        #
        n = int((1 + np.sqrt(8 * len(x_triu) + 2)) / 2)
        x = np.eye(n, dtype=float) * diag

        ij = np.triu_indices(n, 1)
        x[ij] = x_triu
        if symetric:
            x.T[ij] = x[ij]
        else:
            x.T[ij] = -x[ij]
        return x

    @staticmethod
    def stretching_uncertainty(
            cmax: Union[float, np.ndarray], fmin: float, fmax: float, tmin: float, tmax: float) \
            -> Union[float, np.ndarray]:
        """
        Stretching uncertainty after Weaver et al 2011.

        :param cmax: max correlation coefficient from self.corrmax, either a float or a np.ndarray
        :param fmin: lower freq Hz, float
        :param fmax: upper freq Hz, float
        :param tmin: start coda time in s, float
        :param tmax: end coda time in s, float
        :return rmse: uncertainty on epsilon, same type as cmax
        """

        wc = 2. * np.pi * np.sqrt(fmin * fmax)
        T = 1. / (fmax - fmin) / (np.pi * np.sqrt(2.))
        X = cmax
        rmse = np.sqrt(1 - X ** 2.) / (2 * X)
        rmse *= np.sqrt((6 * T * np.sqrt(np.pi / 2.)) / (wc ** 2. * (tmax ** 3 - tmin ** 3)))

        return rmse


class InverseStretcher:
    """
    An object to cancel the effect of the stretching on each trace of a bscan
    This can be used to align the traces with the reference, and then to refine the reference.

    For example:
        you have a bscan of 256 traces with n samples each, bscan is a 2d array shapped (256, n)
        you have an estimate of the stretching history, i.e. 256 epsilon values in an 1D array
        this object returns the bscan corrected from the estimated stretching values
            positive epsilon values (i.e. positive dv/v) mean that the trace was compressed relative to its ref, so this operator stretch it
            negative epsilon values will tend to compress the waveform

    :param t0: time of first sample
    :param nt: number of samples
    :param dt: sampling interval

    :param eps_history: epsilon array, one item per trace in the bscan
    :param interp_kind: which interpolator to use for inverse stretching, among 'linear',

    """
    def __init__(self, t0: float, nt: int, dt: float, eps_history: np.ndarray, interp_kind: Literal['linear'] = "linear"):

        self.t0 = t0
        self.nt = nt
        self.dt = dt
        self.eps_history = eps_history

        self.time_array = self.t0 + np.arange(self.nt) * self.dt
        if interp_kind == "linear":
            block_diagonals = []
            for eps in eps_history:

                op = LinearInterpolator1d(
                    x0=self.t0, nx=self.nt, dx=self.dt, xi=self.time_array * (1. + eps))

                op = op.lininterp_operator.T  # transpose to get the inverser interpolator

                block_diagonals.append(op)
        else:
            raise NotImplementedError(interp_kind)

        self.interpolator = block_diag(block_diagonals, format="csr")

    def __call__(self, bscan: np.ndarray) -> np.ndarray:
        if not bscan.shape == (len(self.eps_history), self.nt):
            raise ValueError(
                f'shape error, bscan is {bscan.shape}'
                f'and should be ({len(self.eps_history)=}, {self.nt=})'
                )
        ans = np.zeros_like(bscan)
        ans.flat[:] = self.interpolator * bscan.flat[:]
        return ans


if __name__ == "__main__":
    from coodddaaaa.utils import Timer, polyspace

    with Timer('constructor'):
        st = Stretcher(
            nt=4096,
            dt=1e-8,
            t0=0.,
            eps=polyspace(-0.01, 0.01, 200, pwr=2.0),
            interp_kind="cubic",
            )

    x = np.random.randn(st.nt)
    with Timer('stretch'):
        x_stretched = st.stretch(x)

    with Timer('corr'):
        c = st.corr(x, x_stretched)

    with Timer('hypermax'):
        epsmax, cmax = st.corrmax(c)

    x = np.random.randn(123, st.nt)
    with Timer('corr_all_with_all'):
        c_triu, e_triu = st.corr_all_with_all(data=x)

    with Timer('triu2dense x2'):
        c = st.triu2dense(c_triu, True, 1.0)
        e = st.triu2dense(e_triu, False, 0.0)

    """
    Timer[constructor]: 312.93 ms
    Timer[stretch]: 12.13 ms
    Timer[corr]: 0.33 ms
    Timer[hypermax]: 0.10 ms
    Timer[corr_all_with_all]: 2904.68 ms
    Timer[triu2dense x2]: 7.41 ms
    """
