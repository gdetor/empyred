# This Python script implements the Simplex Projection and SMAP methods based
# on "Nonlinear forecasting as a way of distinguishing chaos from measurement
# error in time series" by Sugihara and May, 1990. And "Nonlinear forecasting
# for the classification of natural time series" by Sugihara, 1994.
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
import numpy as np
import matplotlib.pylab as plt
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(X, y=0, k=1):
    """
    Computes the Neareser Neigbors. This function is used from Simplex
    Projection and SMAP in order to compute the NNs over the temporal
    embeddings.

    Args:
        X (ndarray): Library of embeddings
        y (ndarray): Target embedding (optional)
        k (int): Number of Nearest Neighbors to compute

    Returns:
        Indices and distances to the kth nearest neighbor
        for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k+1, p=2, radius=0)
    if isinstance(y, np.ndarray):
        knn.fit(X, y)
    else:
        knn.fit(X)
    dist, indices = knn.kneighbors(y.reshape(1, -1), n_neighbors=k+1)
    if dist[0, 0] == 0:
        return indices[:, 1:], dist[:, 1:]
    else:
        # dist, indices = knn.kneighbors(X, n_neighbors=k)
        return indices[:, :-1], dist[:, :-1]


def least_squares(X, y=0):
    """
    Solves the Least Squares problem through SVD.

    Args:
        X (ndarray): Embeddings
        y (ndarray): Target embedding

    Returns:
        Predictions of LSqs as well as the regression coefficient.
    """
    reg = linear_model.LinearRegression()
    if isinstance(y, np.ndarray):
        reg.fit(X, y)
    else:
        reg.fit(X)
    return reg.predict(X), reg.coef_


def create_library(X, E=3, tau=1):
    """
    Builds up the library of the embeddings. It starts with the original input
    signal and delays (rolls) it E (embeddings dimension) times and for tau
    steps in the past.

    Args:
        X (ndarray): Input signal (timeseries)
        E (int): Embeddings dimension
        tau (int): Lag for time delay embeddings

    Returns:
        The library with all the embeddings along with the original signal in
        the placed in the first columnt of the returned array.
    """
    M = X.shape[0]
    tmp = np.zeros((M, E))
    for j in range(E):
        tmp[:, j] = np.roll(X, tau*j)
    return tmp


# def create_library(X, E=3, tau=1):
#     M = X.shape[0]
#     tmp = np.zeros((M-E*tau, E))
#     # tmp = np.zeros((M, E))
#     for j in range(E):
#         tmp[:, j] = np.roll(X, -tau*j)[:-E*tau]
#     return tmp


def simplex_projection(data, library, target, embedding_dim=3, tau=1, Tp=1):
    """
    This function implements the Simplex Projection method proposed by Sugihara
    and May in "Nonlinear forecasting as a way of distinguishing chaos from
    measurement error in time series" (Nature, 1990).

    Args:
        data (ndarray):     Input timeseries
        library (ndarray):  Embeddings library
        target (ndarray):   Target embedding (forecasted)
        embedding_dim (int): Embeddings dimension
        tau (int):          Lag for time delay embeddings
        Tp (int):           Prediction horizon

    Returns:
        A numpy array that contains all the estimated predictions.
    """
    X = data.copy()
    E = embedding_dim
    K = E + 1

    res = library

    index, dists = nearest_neighbors(res, target, k=K)
    index, dist_row = index[0], dists[0]

    min_weight = 1e-6
    min_dist = np.amin(dist_row)
    if min_dist == 0:
        w_dist = np.full(K, min_weight)

        i_dist = np.where(dist_row > 0)
        i_zero = np.where(dist_row == 0)

        if np.size(i_dist):
            w_dist[i_dist] = np.exp(-dist_row[i_dist])

        if np.size(i_zero):
            w_dist[i_zero] = 1
    else:
        w_dist = np.exp(-dist_row / min_dist)
    w = np.fmax(w_dist, min_weight)
    lib_target = np.zeros((K, ))
    for k in range(K):
        idx = index[k] + Tp
        if idx >= res.shape[0]:
            lib_target[k] = X[idx - Tp]
        else:
            lib_target[k] = X[idx]
    y_hat = np.dot(w, lib_target)
    y_hat /= w.sum()
    return y_hat


def smap(data, library, target, lib_size=200, embedding_dim=3,
         n_neighbors=2, tau=1, Tp=1,
         theta=0.01):
    """
    This function implements the SMAP method proposed by Sugihara in "Nonlinear
    forecasting for the classification of natural time series" by Sugihara
    (1994).

    Args:
        data (ndarray):     Input timeseries
        library (ndarray):  Embeddings library
        target (ndarray):   Target embedding (forecasted)
        lib_size (int):     Number of embeddings are being used
        embedding_dim (int): Embeddings dimension
        n_neighbors (int):  Number of nearest neighbors
        tau (int):          Lag for time delay embeddings
        Tp (int):           Prediction horizon
        theta (float):      The nonlinear tuning parameter (see the refered
                            paper)

    Returns:
        A numpy array that contains all the estimated predictions.
    """
    X = data.copy()
    E = embedding_dim
    K = n_neighbors

    X_ = library
    res = library[:lib_size]

    index, dists = nearest_neighbors(res, target, k=K)
    index, dist_row = index[0], dists[0]

    dist_sum = dist_row.mean()
    w = np.zeros((K, ))
    if theta > 0:
        w = np.exp(-theta * dist_row / dist_sum)
    else:
        w = np.ones((K, ))

    A = np.ones((K, E + 1))
    for k in range(K):
        A[k, 0] = w[k]
        for j in range(E):
            A[k, j+1] = w[k] * X_[index[k], j]

    b = np.zeros((K, ))
    for k in range(K):
        idx = index[k] + Tp
        if idx >= res.shape[0]:
            b[k] = X[idx - Tp]
        else:
            b[k] = X[idx]
    b *= w
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=1e-9)
    y_hat = c[0] + np.dot(c[1:], target)
    return y_hat


def compute_error(obs, pred):
    """
    Computes the correlation coefficient (rho), the root mean squared error
    (RMSE) and the mean absolute error (MAE).

    Args:
        obs (ndarray):  Observations (original timeseries)
        pred (ndarray): Predictions

    Returns:
        A tuple of (rho, RMSE, MAE).
    """
    p_not_nan = np.logical_not(np.isnan(pred))
    o_not_nan = np.logical_not(np.isnan(obs))
    i_not_nan = np.intersect1d(np.where(p_not_nan is True),
                               np.where(o_not_nan is True))

    if len(i_not_nan):
        obs = obs[i_not_nan]
        pred = pred[i_not_nan]
    N = len(pred)
    rmse = np.sqrt(np.sum((obs - pred)**2) / N)
    mae = np.sum(np.fabs(obs - pred)) / N

    if sum(pred) == 0:
        rho = 0
    else:
        rho = np.corrcoef(obs, pred)[0, 1]
    return (rho, rmse, mae)


def plot_(X, Y, rho, M=300, U=200):
    """
    Generates all the necessary plots for visual inspection of the results
    obtained from Simplex or SMAP methods.

    Args:
        X (ndarray): Observations
        Y (ndarray): Predictions
        rho (ndarray): Correlation coefficient between observations and
                       predictions
    Returns:
        Void
    """
    idx = np.argmax(np.array(rho))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(Y)):
        ax.plot(Y[i], 'k', zorder=0, alpha=0.3)
    ax.plot(X[U:M+U], 'r', zorder=9)
    ax.plot(Y[idx], 'g', zorder=10)
    ax.set_title("Predictions")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[U:M+U], Y[idx], 'ko', zorder=9, mfc='w')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[U:M+U], 'r', zorder=9)
    ax.plot(Y[idx], 'k', zorder=10)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.array(rho))
    ax.set_xlim([0, 10])
    ax.set_title("rho")
    ax.grid()
