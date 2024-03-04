import numpy as np

def correlation_over_m(ensemble: np.ndarray, m: float):
    ensemble_correlations = ensemble * np.roll(ensemble, -m, 1)
    mean = np.mean(ensemble_correlations)
    std_normalized = np.std(ensemble_correlations) / np.abs(mean) # TODO: calculate std properly

    return mean, std_normalized

def correlation_function(ensemble: np.ndarray, a: float):
    N = int(np.ceil(ensemble.shape[1] / 2)) ## TODO: is this correct?
    mrange = range(N)
    x = np.array(mrange) * a

    correlation, std = np.zeros(N), np.zeros(N)
    for m in mrange:
        correlation[m], std[m] = correlation_over_m(ensemble, m)

    return x, correlation, std

def autocorrelation_estimator(t: int, obs: np.ndarray, obs_mean: float):
    """Eq.(31) from monte carlo errors paper. 'obs' stands for observable"""
    if len(obs.shape) > 1:
        print("ERROR: Observable should be 1d array.")

    # multiply each value at i with the value at i+t, if end of the array is not reached
    obs_correlations = (obs * np.roll(obs, -t))[:-t]

    return np.mean(obs_correlations)

def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y