import numpy as np


def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y
    
def correlation_over_m(ensemble: np.ndarray, m: float):
    ensemble_correlations = ensemble * np.roll(ensemble, -m, 1)
    mean = np.mean(ensemble_correlations)
    std_normalized = np.std(ensemble_correlations) / np.abs(mean) # TODO: calculate std properly

    return mean, std_normalized

def correlation_function(ensemble: np.ndarray, a: float):
    N = int(np.ceil(ensemble.shape[1] / 2)) # TODO: is this correct?
    mrange = range(N)
    x = np.array(mrange) * a

    correlation, std = np.zeros(N), np.zeros(N)
    for m in mrange:
        correlation[m], std[m] = correlation_over_m(ensemble, m)

    return x, correlation, std


def check_1d(array: np.ndarray):
    n = array.ndim
    if n > 1:
        print("ERROR: array should be 1d, but is %id." % n)

def autocorrelation_estimator(t: int, obs: np.ndarray, obs_mean: float):
    """Eq.(31) from monte carlo errors paper. 'obs' stands for observable"""
    check_1d(obs)
    # multiply each value at i with the value at i+t, if end of the array is not reached
    obs_correlations = ((obs-obs_mean) * np.roll((obs-obs_mean), -t))[:-t]

    return np.mean(obs_correlations)

def bin_obs(size: int, obs: np.ndarray):
    """bin size values of obs into one value."""
    check_1d(obs)
    n = len(obs)
    if n % size != 0:
        print("ERROR: array length not divisible by bin size.")

    obs_binned = np.zeros(n // size)
    for i in range(n // size):
        bin_start = i*size
        obs_binned[i] = np.mean(obs[bin_start:bin_start+size])

    return obs_binned