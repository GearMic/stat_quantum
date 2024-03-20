import numpy as np


def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y

def check_nd(array: np.ndarray, n: int):
    """print Error if array is not n-dimensional"""
    m = array.ndim
    if m != n:
        print("ERROR: array should be %id, but is %id." % (n, m))

def autocorrelation_estimator(obs: np.ndarray, t: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    Calculates Eq.(31) from monte carlo errors paper.
    if obs is 2d, does so individually for each row of obs (so along axis 1).
    'obs' stands for observable.
    """

    if obs.ndim == 1:
        obs = np.expand_dims(obs, 0)
    
    if obs_mean == None:
        obs_mean = np.mean(obs, 1)

    deviation = obs - np.tile(obs_mean, (obs.shape[1], 1)).T # deviation from mean for each row
    obs_correlations = ((deviation) * np.roll((deviation), -t, 1)) # multiply each value at i with the value at i+t

    # cut away the values past the end of the array if periodic is false
    if not periodic and t != 0:
        obs_correlations = obs_correlations[:, :-t]

    return np.squeeze(np.mean(obs_correlations, 1))

def autocorrelation_range_mean(obs: np.ndarray, N: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    DEPRECATED
    calculate autocorrelation for t-values up to N.
    if obs is 2d, then the mean of the correlations is taken for each t.
    """

    trange = range(N)

    corr = np.zeros(N)
    for t in trange:
        corr[t] = np.mean(autocorrelation_estimator(obs, t, obs_mean, periodic))

    return np.array(trange), corr

def autocorrelation_range(obs: np.ndarray, N: int, obs_mean: np.ndarray = None, periodic: bool = False):
    """
    calculate autocorrelation for t-values up to N (done along axis 1 if obs if 2d).
    if obs is 2d, then the mean of the correlations is taken for each t.
    """

    if obs.ndim == 1: rows = 1
    else: rows = obs.shape[0]
    trange = range(N)

    corr = np.zeros((rows, N))
    for t in trange:
        corr[:, t] = autocorrelation_estimator(obs, t, obs_mean, periodic)

    return np.array(trange), np.squeeze(corr)

def ensemble_autocorrelation_mean(ensemble: np.ndarray, a: float):
    """
    DEPRECATED
    autocorrelation over an ensemble with lattice distance a and periodic boundary conditions.
    returns array of distance values and array of corresponding correlations.
    """

    N = int(np.ceil(ensemble.shape[1] / 2))
    trange, corr = autocorrelation_range_mean(ensemble, N, periodic=True)

    return trange, corr

def ensemble_autocorrelation(ensemble: np.ndarray, a: float):
    """
    autocorrelation over an ensemble with lattice distance a and periodic boundary conditions.
    returns array of distance values and array of corresponding correlations.
    """

    N = int(np.ceil(ensemble.shape[1] / 2))
    trange, corr = autocorrelation_range(ensemble, N, periodic=True)

    return trange, corr

def bin_mean(obs: np.ndarray, size: int):
    """
    turn size values of obs into one value by calculating the mean.
    """

    check_nd(obs, 1)
    n = len(obs)
    if n % size != 0:
        print("ERROR: array length not divisible by bin size.")

    obs_binned = np.zeros(n // size)
    for i in range(n // size):
        bin_start = i*size
        obs_binned[i] = np.mean(obs[bin_start:bin_start+size])

    return obs_binned