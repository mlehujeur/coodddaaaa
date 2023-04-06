"""
Copyright (c) 2023 maximilien.lehujeur

Linear and cubic interpolation in 1d using fixed grids
and sparse operators for cases where one need 
to interpolate functions on the same grids many times

note I do not use the scipy interpolator
because I need an interpolator that can be created from the grids only
and called later on with the function to interpolate
"""

import matplotlib.pyplot as plt
from scipy import sparse as sp
from scipy.sparse import linalg as splinalg
import numpy as np


class LinearInterpolator1d(object):
    """
    Linear interpolation operator 
    """
    
    def __init__(self, x0: float, nx: int, dx: float, xi: np.ndarray, format: str ="csc"):
        """
        x is the grid at which the function will be defined (nodes)
        xi are the points where the function will be interpolated
        :param x0: x of first sample
        :param nx: number of samples
        :param dx: sampling interval
        :param xi: the points where we need the interpolated values  => f(xi) is computed by self.__call__
        :param format: format to use for the linear operator
        """

        if format == "csc":
            _sp_matrix = sp.csc_matrix
        elif format == "csr":
            _sp_matrix = sp.csr_matrix  # TODO useful?
        else:
            raise ValueError

        self.x0 = x0
        self.nx = nx
        self.dx = dx
        self.xi = xi

        ni = len(xi)  # numb of interp poinst
        x = np.arange(nx) * dx + x0  # nodes
        xmin, xmax = x.min(), x.max()  # interp bounds

        # find indexs of x of nodes located after the interp points xi
        k = np.ceil((xi - xmin) / dx).astype(int)

        # find the interp points that occur within the interp bounds
        m = (k > 0) & (k < nx)  # mask
        m = np.arange(len(xi))[m]  # to indexs
        km = k[m]  # eliminate the interp points out of bounds => interp will return 0 for these nodes
        xim = xi[m]  # same

        # nodes before the interpolation points
        rows1 = m
        cols1 = km - 1
        vals1 = (x[km] - xim) / self.dx

        # nodes after the interpolation points
        rows2 = m
        cols2 = km
        vals2 = (xim - x[km-1]) / self.dx

        # assemble
        self.operator = _sp_matrix(
            (np.concatenate((vals1, vals2)),
             (np.concatenate((rows1, rows2)),
              np.concatenate((cols1, cols2)))
             ),
            shape=(ni, nx))

    def __call__(self, f: np.ndarray):
        """
        affect function values at x and return the interpolated values at xi
        """
        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == len(self.x)

        return self.operator * f


class SecondDerivativeOperatorTypeII:
    def __init__(self, x0: float, nx: int, dx: float, format: str ="csc"):
        """
        Second derivative operator order 3 in the internal domain,
        Implement type II boundary condition after https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
        For a regular grid only
        x is the grid at which the function will be defined (nodes)
        xi are the points where the function will be interpolated
        :param x0: x of first sample
        :param nx: number of samples
        :param dx: sampling interval
        :param format: format of the sparse operator
        """
        self.x = x0 + np.arange(nx) * dx

        if format == "csc":
            _sp_matrix = sp.csc_matrix
        elif format == "csr":
            _sp_matrix = sp.csr_matrix
        else:
            raise ValueError

        idx2 = dx ** -2.

        # do not fill rows 0 and nx - 1
        # for type II boundary condition
        # d[0] = 2 * f''0 = 0
        # d[-1] = 2 * f''[-1] = 0
        diag = np.zeros(nx)
        diag[1:-1] = -2. * idx2

        upper_diag = np.zeros(nx - 1)
        upper_diag[1:] = idx2

        lower_diag = np.zeros(nx - 1)
        lower_diag[:-1] = idx2
        self.operator = sp.diags(
            (lower_diag, diag, upper_diag),
            offsets=(-1, 0, 1),
            shape=(nx, nx),
            format=format,
            )

    def __call__(self, f: np.ndarray):
        """
        compute the derivative of f on x
        """
        # assert isinstance(f, np.ndarray)
        # assert f.ndim == 1
        # assert len(f) == len(self.x)

        return self.operator * f


class CubicInterpolator1d:
    def __init__(self, x0: float, nx: int, dx: float, xi: np.ndarray, format: str ="csc"):
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

        # ==== set the grids
        self.x0, self.nx, self.dx = x0, nx, dx
        self.x = x0 + np.arange(nx) * dx  # nodes
        self.xi = xi  # interp points

        # ==== build the internal operators
        self.linear_interpolator = LinearInterpolator1d(x0=x0, nx=nx, dx=dx, xi=xi, format=format)
        self.second_derivativor = SecondDerivativeOperatorTypeII(x0=x0, nx=nx, dx=dx, format=format)
        # multiply by 3 because 6*f[xi-1,xi,xi+1] means 3 * [f(xi+1) - 2 * f(xi) + f(xi-1)] / (hi**2)
        self.second_derivativor.operator = 3. * self.second_derivativor.operator

        # left term in eq (6) from https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
        upper_diag = .5 * np.ones(nx-1, float)  # lambda terms
        diag = 2 * np.ones(nx, float)   # diagonal terms
        lower_diag = 0.5 * np.ones(nx-1, float)  # mu terms
        lower_diag[-1] = upper_diag[0] = 0.  # type II boundary condition
        a = sp.diags((lower_diag, diag, upper_diag), offsets=(-1, 0, 1), format=format)
        # inverse_of_a = splinalg.inv(a)  #=> dense
        self.solver = splinalg.splu(a)  # TODO : CORRECT? / OPTIMAL?

        # ==== Implement the operator for equation (1), with respect to Mis coefficients
        #      the missing terms for yi and yi+1 are included in the linear_operator term
        if format == "csc":
            _sp_matrix = sp.csc_matrix
        elif format == "csr":
            _sp_matrix = sp.csr_matrix  # Useful?
        else:
            raise ValueError

        ni = len(xi)  # numb of interp poinst
        x = np.arange(nx) * dx + x0  # nodes
        xmin, xmax = x.min(), x.max()  # interp bounds

        # find indexs of x of nodes located after the interp points xi
        k = np.ceil((xi - xmin) / dx).astype(int)

        # find the interp points that occur within the interp bounds
        m = (k > 0) & (k < nx)  # mask
        m = np.arange(len(xi))[m]  # to indexs
        km = k[m]  # eliminate the interp points out of bounds => interp will return 0 for these nodes
        xim = xi[m]  # same

        # nodes before the interpolation points
        rows1 = m
        cols1 = km - 1
        vals1 = (x[km] - xim)**3. / (6. * self.dx) - self.dx * (x[km] - xim) / 6.

        # nodes after the interpolation points
        rows2 = m
        cols2 = km
        vals2 = (xim - x[km-1])**3. / (6. * self.dx) - self.dx * (xim - x[km-1]) / 6.

        # assemble
        self.eq1_operator = _sp_matrix(
            (np.concatenate((vals1, vals2)),
             (np.concatenate((rows1, rows2)),
              np.concatenate((cols1, cols2)))
             ),
            shape=(ni, nx))

    def __call__(self, f):
        assert isinstance(f, np.ndarray)
        assert f.ndim == 1
        assert len(f) == len(self.x)

        # 3 * f[xi-1, xi, xi+1], d0=d-1=0 for type II boundary condition
        d = self.second_derivativor.operator * f

        # solve equation (6) for M
        m = self.solver.solve(d)

        return self.eq1_operator * m + self.linear_interpolator.operator * f

    # TODO : add method to export/reload from npz file


if __name__ == "__main__":

    x = np.linspace(0.1, 0.9, 10)
    xi = np.linspace(-0.1, 1.1, 1000)

    P = LinearInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    C = CubicInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)

    f = np.random.randn(len(x))

    pi = P(f)
    ci = C(f)

    if True:
        plt.figure()
        plt.plot(x, f, "ko")
        plt.plot(xi, pi, 'r-', label='linear')
        plt.plot(xi, ci, 'g-', label='cubic')
        plt.gca().legend()
        plt.show()

    # ========================= stretcher
    x = np.linspace(0.1, 0.9, 100)
    xi = x * (1. + 0.01)

    P = LinearInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)
    C = CubicInterpolator1d(x0=x[0], nx=len(x), dx=x[1] - x[0], xi=xi)

    f = np.sin(2. * np.pi * x / 0.1)

    pi = P(f)
    ci = C(f)

    if True:
        plt.figure()
        plt.plot(x, f, "k")
        plt.plot(x, pi, 'r-', label='linear')
        plt.plot(x, ci, 'g-', label='cubic')
        plt.gca().legend()
        plt.show()
