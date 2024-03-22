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
    # if n % size != 0:
    #     print("ERROR: array length not divisible by bin size.")
    # n_bins = n // size
    n_bins = np.floor_divide(n, size) 

    obs_binned = np.zeros(n_bins)
    for i in range(n_bins):
        bin_start = i*size
        obs_binned[i] = np.mean(obs[bin_start:bin_start+size])

    return obs_binned

def errors_of_binned(obs: np.ndarray, max_size: int):
    """
    calculates errors of obs for increasing bin size.
    Error calculated as error of mean.
    """

    # calculate standard deviation for different bin sizes
    binsize, error = [i+1 for i in range(max_size)], []
    for size in binsize:
        obs_binned = bin_mean(obs, size)
        error.append(np.std(obs_binned)/len(obs_binned))

    return binsize, error


def bin_mean_error_plot_array(fig, ax, observables: np.ndarray):
    """
    uses data: 2d array with different observables along axis 1,
    and different draws of each observable on axis 0.
    plots errors for increasing bin size. Does so for each observable in the same plot.
    """

    irange = range(observables.shape[1])
    for i in irange:
        obs = observables[:, i]
    
"""         # calculate standard deviation for different bin sizes
        bin_plot_xrange = (0, 40)
        bin_plot_yrange = (-0.1, 0.5)
        fig, ax = plt.subplots()

        binsize, error = [], []
        for i in range(n_bin_steps+1):
            binsize.append(2**i)
            error.append(np.std(obs)/len(obs))

            # next bin size
            obs = bin_mean(obs, 2)

        ax.set_xlim(bin_plot_xrange)
        ax.set_ylim(bin_plot_yrange)
        ax.set_xlabel("Monte carlo time")
        ax.set_ylabel("Correlation")
        ax.set_title("Correlation for bin sizes " + str(binsize[:n_bin_plots]))
        ax.grid()
        ax.minorticks_on()
        fig.savefig("plot/B_correlation.png", dpi=dpi)
    
        # plot naive error for the bin sizes
        fig, ax = plt.subplots()
        ax.plot(binsize, error)
        ax.set_xlabel("Bin size")
        ax.set_ylabel("Error")
        fig.savefig('plot/B_error.png', dpi=dpi) """
