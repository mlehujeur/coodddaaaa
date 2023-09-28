"""
Copyright (c) 2023 maximilien.lehujeur

Linear and cubic interpolation in 1d using fixed grids
and sparse operators for cases where one need 
to interpolate functions on the same grids many times

note I do not use the scipy interpolator
because I need an interpolator that can be created from the grids only
and called later on with the function to interpolate

2023.04.07 : P. Mora : Speed up the construction of the sparse matrixes => x20 to x50
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as splinalg
from scipy.fftpack import rfft  # NOT scipy.fft.rfft !!!

SPMATRIXFORMAT = {"csc": sp.csc_matrix, "csr": sp.csr_matrix}


class LinearInterpolator1d:
    """
    Linear interpolation operator

    :param x0: x coordinate of the first sample
    :param nx: number of samples
    :param dx: sampling interval
    :param xi: the points where we need the interpolated values
        => f(xi) is computed by self.__call__
    :param format: format to use for the linear operator
    """

    def __init__(
            self,
            x0: float, nx: int, dx: float,
            xi: np.ndarray,
            format: str ="csc"):
        """
        x is the grid at which the function will be defined (nodes)
        xi are the points where the function will be interpolated
        """

        self.x0 = x0
        self.nx = nx
        self.dx = dx
        self.xi = xi

        # nodes
        x = np.arange(nx) * dx + x0

        self._idx_xi_to_interp, self._idx_x_after_interp_points = \
            self.find_interp_points_in_grid(x0, nx, dx, xi)

        # nodes before the interpolation points
        rows1 = self._idx_xi_to_interp
        cols1 = self._idx_x_after_interp_points - 1
        dxafter = (x[self._idx_x_after_interp_points] - xi[self._idx_xi_to_interp])
        vals1 = dxafter / self.dx

        # nodes after the interpolation points
        rows2 = rows1  # self._idx_xi_to_interp
        cols2 = self._idx_x_after_interp_points
        vals2 = 1. - vals1  # dxbefore / self.dx

        rows = np.concatenate((rows1, rows2))
        cols = np.concatenate((cols1, cols2))
        vals = np.concatenate((vals1, vals2))

        # assemble
        self.lininterp_operator = \
            SPMATRIXFORMAT[format](
                (vals, (rows, cols)),
                shape=(len(self.xi), self.nx))
   
    def find_interp_points_in_grid(self, x0: float, nx: int, dx: float, xi: np.ndarray):

        # find indexs of x of nodes located after the interp points xi
        k = np.ceil((xi - x0) / dx).astype(int)

        # find the interp points that occur within the interp bounds
        m = (k > 0) & (k < nx)  # mask
        idx_xi_to_interp = np.arange(len(xi))[m]  # to indexs

        # eliminate the interp points out of bounds
        # => interp will return 0 for these points
        idx_x_after_interp_points = k[idx_xi_to_interp]

        return idx_xi_to_interp, idx_x_after_interp_points

    def __call__(self, f: np.ndarray):
        """
        affect function values at x and return the interpolated values at xi
        """
        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == len(self.x)

        return self.lininterp_operator * f


class SecondDerivativeOperatorTypeII:
    """
    Second derivative operator order 3 in the internal domain,
    Implement type II boundary condition after https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
    For a regular grid only
    x is the grid at which the function will be defined (nodes)
    xi are the points where the function will be interpolated

    :param nx: number of nodes
    :param dx: sampling interval between nodes
    :param format: format of the sparse operator
    """

    def __init__(self, nx: int, dx: float, format: str ="csc"):

        idx2 = dx ** -2.
        self.operator = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx), format=format) * idx2
        # for type II boundary condition => d[0] = 2 * f''0 = 0
        self.operator[0, 0] = 0
        self.operator[0, 1] = 0
        # for type II boundary condition => d[-1] = 2 * f''[-1] = 0
        self.operator[-1, -1] = 0
        self.operator[-1, -2] = 0

    def __call__(self, f: np.ndarray):
        """
        compute the derivative of f on x
        """
        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == len(self.x)

        return self.operator * f


class CubicInterpolator1d(LinearInterpolator1d):
    """
    Lagrange Cubic interpolation with boundary type II from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
    Works only on a regular grid for now
    x is the grid at which the function will be defined (nodes)
    xi are the points where the function will be interpolated
    :param x0: x of first sample
    :param nx: number of samples
    :param dx: sampling interval
    :param xi: array of points where to interpolate the function
    :param format: format of the sparse operator
    """

    def __init__(
            self,
            x0: float, nx: int, dx: float,
            xi: np.ndarray,
            format: str ="csc"):

        LinearInterpolator1d.__init__(self, x0=x0, nx=nx, dx=dx, xi=xi, format=format)
        _sp_matrix = SPMATRIXFORMAT[format]

        # ==== add more internal operators to move to cubic interpolation
        # multiply by 3 because 6*f[xi-1,xi,xi+1] means 3 * [f(xi+1) - 2 * f(xi) + f(xi-1)] / (hi**2)
        self.derivator = 3. * SecondDerivativeOperatorTypeII(nx=nx, dx=dx, format=format).operator

        # left term in eq (6) from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
        upper_diag = .5 * np.ones(nx-1, float)  # lambda terms
        diag = 2 * np.ones(nx, float)   # diagonal terms
        lower_diag = 0.5 * np.ones(nx-1, float)  # mu terms
        lower_diag[-1] = upper_diag[0] = 0.  # type II boundary condition
        a = sp.diags((lower_diag, diag, upper_diag), offsets=(-1, 0, 1), format=format)
        # inverse_of_a = splinalg.inv(a)  #=> dense
        self.solver = splinalg.splu(a)  # => time cost is negligible (0.4%)

        # ==== Implement the operator for equation (1), with respect to Mis coefficients
        #      the missing terms for yi and yi+1 are included in the self.lininterp_operator term
        x = np.arange(nx) * dx + x0

        # nodes before the interpolation points
        rows1 = self._idx_xi_to_interp
        cols1 = self._idx_x_after_interp_points - 1
        dxafter = (x[self._idx_x_after_interp_points] - xi[self._idx_xi_to_interp])
        vals1 = dxafter ** 3. / (6. * self.dx) - self.dx * dxafter / 6.

        # nodes after the interpolation points
        rows2 = self._idx_xi_to_interp
        cols2 = self._idx_x_after_interp_points
        dxbefore = (xi[self._idx_xi_to_interp] - x[self._idx_x_after_interp_points-1])
        vals2 = dxbefore ** 3. / (6. * self.dx) - self.dx * dxbefore / 6.

        rows = np.concatenate((rows1, rows2))  # np.repeat(r_, 2)
        cols = np.concatenate((cols1, cols2))  # np.array([k_-1, k_]).T.flatten()
        vals = np.concatenate((vals1, vals2))  # np.array([v1, 1-v1]).T.flatten()

        # assemble
        self.cubinterp_operator = \
            SPMATRIXFORMAT[format](
                (vals, (rows, cols)),
                shape=(len(self.xi), self.nx))

    def __call__(self, f):

        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == self.nx

        # 3 * f[xi-1, xi, xi+1], d0=d-1=0 for type II boundary condition
        d = self.derivator * f

        # solve equation (6) for M
        m = self.solver.solve(d)

        return self.cubinterp_operator * m + self.lininterp_operator * f

    # TODO : add method to export/reload from npz file


class RFFTInterpolator1d:
    """
    Fourier Interpolator based on rfft

    :param x0: x of first sample
    :param nx: number of samples
    :param dx: sampling interval
    :param xi: array of points where to interpolate the function
    :param format: format of the sparse operator

    """
    def __init__(
            self, 
            x0: float, nx: int, dx: float,
            xi: np.ndarray):

        self.x0 = x0
        self.nx = nx
        self.dx = dx
        self.xi = xi

        npts = nx
        fnfft = npts
        two_pi = 2.0 * np.pi

        n = (xi - x0) / ((nx - 1) * dx) * (npts - 1)

        k = np.arange(npts)[:, np.newaxis]
        b = np.zeros((npts, len(xi)))
        b[0, :] = 1.0

        if npts % 2:
            b[1:npts:2] = +2.0 * np.cos(two_pi * n * k[1:(npts - 1) // 2 + 1, :] / fnfft)
            b[2:npts:2] = -2.0 * np.sin(two_pi * n * k[1:(npts - 1) // 2 + 1, :] / fnfft)

        else:
            b[1:npts - 1:2] = +2.0 * np.cos(two_pi * n * k[1:npts // 2, :] / fnfft)
            b[2:npts - 1:2] = -2.0 * np.sin(two_pi * n * k[1:npts // 2, :] / fnfft)
            b[-1, :] = +1.0 * np.cos(two_pi * n * npts / fnfft / 2.)

        b *= 1.0 / fnfft
        self._irfft_basis = b

    def __call__(self, f: np.ndarray):
        """
        affect function values at x and return the interpolated values at xi
        """
        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == len(self.x)

        return rfft(f, axis=-1).dot(self._irfft_basis)


if __name__ == "__main__":

    # ========================= simple interpolation test
    # build the grids
    x = np.linspace(0.1, 0.9, 10)  # the nodes
    xi = np.linspace(-0.1, 1.1, 1000)  # the interpolation points

    # build the linear operators
    P = LinearInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    C = CubicInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    F = RFFTInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    # pick some random values to attach to the nodes
    f = np.random.randn(len(x))

    # call the operators for these values
    pi = P(f)
    ci = C(f)
    fi = F(f)

    # compare the output
    if True:
        plt.figure()
        plt.plot(x, f, "ko")
        plt.plot(xi, pi, 'r-', label='linear')
        plt.plot(xi, ci, 'g-', label='cubic')
        plt.plot(xi, fi, 'm-', label='Fourier')
        plt.gca().legend()
        plt.show()

    # ========================= interpolation for stretching
    x = np.linspace(0.1, 0.9, 100)
    xi = x * (1. + 0.01)

    P = LinearInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    C = CubicInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    F = RFFTInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)

    f = np.sin(2. * np.pi * x / 0.1)

    pi = P(f)
    ci = C(f)
    fi = F(f)

    if True:
        plt.figure()
        plt.plot(x, f, "k")
        plt.plot(x, pi, 'r-', label='linear')
        plt.plot(x, ci, 'g-', label='cubic')
        plt.plot(x, fi, 'm-', label='Fourier')
        plt.gca().legend()
        plt.show()

    # ========================= interpolation of many signals at ones
    x = np.linspace(0.1, 0.9, 10)  # the nodes
    xi = np.linspace(-0.1, 1.1, 1000)  # the interpolation points

    P = LinearInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    C = CubicInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    F = RFFTInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)

    f = 0.2 * np.random.randn(10, len(x))
    pi = P(f.T).T
    ci = C(f.T).T
    fi = F(f)   # WARNING : Fourier interp is not built the same way as the two other interpolators
    print(ci.shape)
    print(fi.shape)

    if True:
        plt.figure()
        for n in range(f.shape[0]):
            plt.plot(x, f[n, :] + n, "ko")
            plt.plot(xi, pi[n, :] + n, 'r-', label='linear' if not n else None)
            plt.plot(xi, ci[n, :] + n, 'g-', label='cubic' if not n else None)
            plt.plot(xi, fi[n, :] + n, 'm-', label='Fourier' if not n else None)
        plt.gca().legend()
        plt.show()
