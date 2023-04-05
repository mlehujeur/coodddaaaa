import numpy as np
from coodddaaaa.interp1d import LinearInterpolator1d, CubicInterpolator1d
from coodddaaaa.hypermax import hypermax


class Stretcher:
    """
    An object to compute stretched signals and stretching correlation as defined in Weaver et al., 2011
    The object pre-computes the interpolation operator
    The user can pre-compute and store the stretched basis of the reference signal
    This basis can then be provided for stretching correlation with a new signal
    This object can also compute the stretching between all pairs of signals in a b-scan
    """
    def __init__(self,
                 t0: float, dt: float, nt: int,
                 eps: np.ndarray,
                 norm: bool = False,
                 interp_kind: str = "cubic"):
        """
        :param t0: time of first sample
        :param dt: sampling interval
        :param nt: number of samples
        :param eps: epsilon array
        :param norm: use it to compute normalized correlation.
                     warning : for stretching only, use norm = False
        :param interp_kind: which interpolator to use for stretching
        """
        assert interp_kind in ['linear', 'cubic']

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
                x=self.t,  # nodes
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
        else:
            raise ValueError(interp_kind)

    def stretch(self, x: np.ndarray) -> np.ndarray:
        """
        compute the stretched basis functions from a signal x
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
            norm = (x_stretched * x_stretched).sum(axis=1) ** -0.5
            # norm = np.linalg.norm(x_stretched, axis=1)

            x_stretched *= norm[:, np.newaxis]

        return x_stretched

    def corr(self, x: np.ndarray, x_stretched: np.ndarray) -> np.ndarray:
        """
        Stretching correlation of x with a basis of stretched versions of the reference signal
        :param x: signal to be correlated, np.ndarray, 1d, shape (nt, )
        :param x_stretched: stretched reference from self.stretch, np.ndarray, 2d, shape (neps, nt, )
        """

        c = x_stretched.dot(x) * self.dt

        if self.norm:
            # the normalization relative to x_stretched
            # is already done
            c /= x.dot(x) ** 0.5 * self.dt

        return c

    def corrmax(self, c: np.ndarray) -> (float, float):
        """
        find the maximum of the correlation function with subsample precision
        """
        epsmax = hypermax(
            time_array=self.eps, function_array=c,
            assume_t_growing=True)
        return epsmax, c.max()

    def corr_all_with_all(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        correlate all possible pairs of signals in a bscan
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
        itriu, jtriu = np.triu_indices(ntraces, 1)

        n = 0
        for i in range(ntraces - 1):
            y = data[i, :]
            y_stretched = self.stretch(y)

            for j in range(i + 1, ntraces):
                x = data[j, :]
                cij = self.corr(x, y_stretched)
                emax, cmax = self.corrmax(cij)

                c_triu[n] = cmax
                e_triu[n] = emax
                n += 1

        return c_triu, e_triu

    @staticmethod
    def triu2dense(x_triu, symetric: bool, diag: float) -> np.ndarray:
        """
        convert upper triangle matrix to square matrix
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
    def stretching_uncertainty(c, fmin, fmax, t1, t2):
        """
        stretching uncertainty after Weaver et al 2011
        :param c: correlation coefficient
        :param fmin: lower freq Hz
        :param fmax: upper freq Hz
        :param t1: start coda time in s
        :param t2: end coda time in s
        :return rmse: uncertainty on epsilon
        """
        wc = 2. * np.pi * np.sqrt(fmin * fmax)
        T = 1. / (fmax - fmin) / (np.pi * np.sqrt(2.))
        X = c
        rmse = np.sqrt(1 - X ** 2.) / (2 * X)
        rmse *= np.sqrt((6 * T * np.sqrt(np.pi / 2.)) / (wc ** 2. * (t2 ** 3 - t1 ** 3)))
        # print(rmse.min(), rmse.max())
        return rmse
