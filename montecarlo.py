import numpy as np


def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y
    
# def correlation_over_m(ensemble: np.ndarray, m: float):
#     ensemble_correlations = ensemble * np.roll(ensemble, -m, 1)
#     mean = np.mean(ensemble_correlations)
#     std_normalized = np.std(ensemble_correlations) / np.abs(mean) # TODO: calculate std properly

#     return mean, std_normalized

# def correlation_function(ensemble: np.ndarray, a: float): # TODO: replace this by autocorrelation_estimator
#     N = int(np.ceil(ensemble.shape[1] / 2)) # TODO: is this correct?
#     mrange = range(N)
#     x = np.array(mrange) * a

#     correlation, std = np.zeros(N), np.zeros(N)
#     for m in mrange:
#         correlation[m], std[m] = correlation_over_m(ensemble, m)

#     return x, correlation, std


def check_nd(array: np.ndarray, n: int):
    """print Error if array is not n-dimensional"""
    m = array.ndim
    if m != n:
        print("ERROR: array should be %id, but is %id." % (n, m))

def autocorrelation_estimator(t: int, obs: np.ndarray, obs_mean: np.ndarray, periodic: bool = False):
    """
    Calculates Eq.(31) from monte carlo errors paper.
    if obs is 2d, does so individually for each row of obs.
    'obs' stands for observable.
    """

    if obs.ndim == 1:
        obs = np.expand_dims(obs, 0)

    deviation = obs - np.tile(obs_mean, (obs.shape[1], 1)).T # deviation from mean for each row
    obs_correlations = ((deviation) * np.roll((deviation), -t, 1)) # multiply each value at i with the value at i+t

    # cut away the values past the end of the array if periodic is false
    if not periodic:
        obs_correlations = obs_correlations[:, :-t]

    return np.squeeze(np.mean(obs_correlations, 1))

def correlation_function(ensemble: np.ndarray, a: float):
    """
    autocorrelation over an ensemble with lattice distance a and periodic boundary conditions.
    returns array of distance values and corresponding correlations
    """
    N = int(np.ceil(ensemble.shape[1] / 2)) # max amount of t values without repetitions
    trange = range(N)
    distance = np.array(trange) * a

    means = np.mean(ensemble, 1)
    correlation = np.zeros(N)
    for t in trange:
        correlation[t] = np.mean(autocorrelation_estimator(t, ensemble, means, periodic=True))

    return distance, correlation

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