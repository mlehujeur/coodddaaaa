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
    
    def __init__(self, x: np.ndarray, xi: np.ndarray, format: str ="csc"):
        """
        :param x: the points where the function is defined (nodes)  => f(x) is provided in self.__call__
        :param xi: the points where we need the interpolated values  => f(xi) is computed by self.__call__
        :param format: format to use for the linear operator
        """
        # TODO : go back to a regular grid, take x0, nx, dx in arguments to make sure the user is aware of this constraint
        assert x.ndim == 1
        assert (x[1:] > x[:-1]).all()
        assert xi.ndim == 1
        # assert xi.min() > x.min()
        # assert xi.max() < x.max()

        if format == "csc":
            _sp_matrix = sp.csc_matrix
        elif format == "csr":
            _sp_matrix = sp.csr_matrix  # TODO useful?
        else:
            raise ValueError

        self.x = x
        self.xi = xi
        rows = []
        cols = []
        vals = []
        xmin, xmax = x.min(), x.max()
        for i in range(len(self.xi)):
            if xmin < self.xi[i] < xmax:
                # TODO : avoid searchorted
                k = np.searchsorted(self.x, self.xi[i], side="right")
                assert self.x[k-1] <= self.xi[i] < self.x[k]

                hk = (self.x[k] - self.x[k - 1])  # width of the segment
                x = self.xi[i]  # evaluation point

                rows.append(i)
                cols.append(k-1)
                vals.append((self.x[k] - x) / hk)

                rows.append(i)
                cols.append(k)
                vals.append((x - self.x[k-1]) / hk)
            else:
                # otherwise the interpolation will be 0, add a fictive row in the linear operator
                pass

        self.operator = _sp_matrix((vals, (rows, cols)), shape=(len(self.xi), len(self.x)))
        
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
        Works only on a regular grid
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
        self.x = x0 + np.arange(nx) * dx  # nodes
        self.xi = xi  # interp points

        # ==== build the internal operators
        self.linear_interpolator = LinearInterpolator1d(x=self.x, xi=xi, format=format)
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

        rows = []
        cols = []
        vals = []

        xmin, xmax = self.x.min(), self.x.max()
        for i in range(len(self.xi)):
            if xmin < self.xi[i] < xmax:
                # TODO : avoid searchorted, the grid is regular!
                k = np.searchsorted(self.x, self.xi[i], side="right")
                assert self.x[k-1] <= self.xi[i] < self.x[k]
                hk = (self.x[k] - self.x[k-1])
                assert hk > 0

                x = self.xi[i]  # evaluation point

                rows.append(i)
                cols.append(k-1)
                vals.append((self.x[k] - x)**3. / (6. * hk) - hk * (self.x[k] - x) / 6.)

                rows.append(i)
                cols.append(k)
                vals.append((x - self.x[k-1])**3. / (6. * hk) - hk * (x - self.x[k-1]) / 6.)
            else:
                # otherwise the interpolation will be 0, add a fictive row in the linear operator
                pass

        self.eq1_operator = _sp_matrix((vals, (rows, cols)), shape=(len(self.xi), len(self.x)))

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
    xi = np.linspace(0., 1.0, 1000)

    P = LinearInterpolator1d(x=x, xi=xi)
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

    P = LinearInterpolator1d(x=x, xi=xi)
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
