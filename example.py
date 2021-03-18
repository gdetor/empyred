# This script demonstrates how the Simplex Projection and SMAP methods work.
# Copyright (C) 2019  Georgios Is. Detorakis (gdetor@protonmail.com)
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
import sys
import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler

from empyred import create_library, compute_error, plot_
from empyred import simplex_projection, smap


def lorenz(X, t, sigma, beta, rho):
    """
    Lorenz dynamical system. This functions is meant to be used along with
    Scipy's odeint method.

    Args:
        X (ndarray): Dynamica system states (3 states)
        t (float): Time step
        sigma (float):  Prandtl number
        beta (float):   Physical dimensios of the system itself
        rho (float):    Rayleigh number

    Returns:
        Numpy array with the current state (prepared for odeint of scipy).
    """
    x = X[0]
    y = X[1]
    z = X[2]

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


if __name__ == '__main__':
    signal = 'tent'     # Signal upon the test runs

    # Different cases can be used
    if signal == 'sin':         # Sinusoidal
        t = np.linspace(0, 1, 1000)
        X_ = np.sin(2.*np.pi*t*10) + np.random.normal(0, .05, (1000,))
    elif signal == 'tent':      # Tent map
        X_ = np.genfromtxt('./data/tentmapr.dat')
        # scaler = MinMaxScaler(feature_range=(-1, 1), copy=False)
        # X_ = scaler.fit_transform(X_.reshape(-1, 1))[:, 0]
    elif signal == 'lorenz':    # Lorenz system
        x0 = np.array([-8.0, 8.0, 27.0])
        t = np.linspace(0, 200, 10000)
        sol = odeint(lorenz, x0, t, (10.0, 8./3., 28.0),)
        scaler = MinMaxScaler(copy=False)
        sol = scaler.fit_transform(sol)
        X_ = sol[:3000, 0].copy()
    elif signal == '3sp':   # 3 species system
        x = np.genfromtxt("./data/x_3sp")
        X_ = x
        y = np.genfromtxt("./data/x_3sp")
        xy = np.hstack([x, y])
    else:
        print("No input signal!")
        sys.exit(-1)

    print("Identify optimal embedding dimension.")
    print("================================================")
    M, U, L, tau = 300, 200, 100, 1
    rho_, Y_ = [], []
    for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        partial = tau * (e - 1)
        X = X_[partial:X_.shape[0] - partial].copy()
        res = create_library(X, E=e, tau=1)
        Y = np.zeros((M, ))
        Y[0] = X[U]
        for i in range(M-1):
            Y[i+1] = simplex_projection(X, res[:L], target=res[U+i],
                                        embedding_dim=e, tau=1, Tp=1)
        Y_.append(Y)
        rho, rmse, mae = compute_error(X[U:M+U], Y)
        print(rho, rmse, mae)
        rho_.append(rho)
    opt = np.argmax(rho_) + 1
    plot_(X, Y_, rho_)

    print("Examine prediction decay.")
    print("================================================")
    M, U, L = 300, 200, 100
    rho_, Y_ = [], []
    res = create_library(X_.copy(), E=opt, tau=1)
    for tt in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        X = X_.copy()
        Y = np.zeros((M, ))
        Y[0] = X[U]
        for i in range(M-tt):
            Y[i+tt] = simplex_projection(X, res[:L], target=res[U+i],
                                         embedding_dim=opt, tau=1, Tp=tt)
        Y_.append(Y)
        rho, rmse, mae = compute_error(X[U:M+U], Y)
        print(rho, rmse, mae)
        rho_.append(rho)
    plot_(X, Y_, rho_)

    print("Identify Nonlineariry.")
    print("================================================")

    M, U, L = 300, 200, 1000
    rho_, Y_ = [], []
    res = create_library(X_.copy(), E=2, tau=1)
    for th in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        X = X_.copy()
        Y = np.zeros((M, ))
        Y[0] = X[U]
        for i in range(M-1):
            Y[i+1] = smap(X, res, target=res[U+i], lib_size=L,
                          embedding_dim=2, theta=th, n_neighbors=998)
        Y_.append(Y)
        rho, rmse, mae = compute_error(X[U:M+U], Y)
        rho_.append(rho)
        print(rho, rmse, mae)
    plot_(X, Y_, rho_)
    plt.show()
