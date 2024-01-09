#### for plotting

import numpy as np
import matplotlib.pyplot as plt
import os.path


def list_filename(name, suffix='.png'): # to save plots successively
    i = 0
    while True:
        filename = '%s%i%s' % (name, i, suffix)
        if os.path.isfile(filename):
            i += 1
            continue

        return filename


def correlation_function(ensemble: np.ndarray, m: int, a):
    rows, cols = ensemble.shape

    correlation_sum = 0
    for i in range(rows):
        row = ensemble[i]
        for j in range(cols-m):
            correlation_sum += row[j] * row[j+m]

    return 1 / rows / (cols-m) * correlation_sum


def correlation_function_alt(ensemble: np.ndarray, m: int, a: float, j: int = 0):
    rows, cols = ensemble.shape

    correlation_sum = 0

    for i in range(rows):
        row = ensemble[i]
        correlation_sum += row[j] * row[j+m]

    return 1 / rows * correlation_sum


def correlation_function_periodic(ensemble: np.ndarray, m: int, a: float):
    # ensemble = ensemble[:, :-1] # cut away duplicate periodic value
    data = ensemble.flatten()

    correlation_sum = 0
    M = len(data)
    for i in range(M):
        correlation_sum += data[i%M] * data[(i+m)%M]

    return 1 / M * correlation_sum
    
    
def bin_normalized(data, n_bins, xlower, xupper):
    rows, cols = data.shape
    bin_size = (xupper - xlower) / n_bins

    bins, edges = np.histogram(data, n_bins, (xlower, xupper))
    bins_x = edges[:-1] + bin_size / 2
    bins_y = bins / bin_size / (rows*cols)

    return bins_x, bins_y



dpi = 200
ms = 4
# ax = plt.gca()


## Fig.4 (more or less)
measurements = np.genfromtxt('harmonic_a.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
rows, cols = measurements.shape
t = range(cols)

for row in range(3):
    # plt.plot(t, measurements[row])
    plt.plot(t, measurements[-row-1])

plt.savefig(list_filename('plot/4/plot'), dpi=dpi)
# plt.savefig('plot/4/ztest.png', dpi=dpi)


## Fig. 5
# bins = np.genfromtxt('bins.csv', delimiter=',')
# bins_x, bins_y = bins[0], bins[1]
# bin_size = bins_x[1] - bins_x[0]
# bins_x += bin_size / 2 # for plotting bin value in the middle of each bin

data = measurements

xlower, xupper = -3., 3.
bins_x, bins_y = bin_normalized(data, 40, xlower, xupper)

# func_x = np.linspace(bins_x[0], bins_x[-1] + bin_size, 100)
func_x = np.linspace(xlower, xupper, 200)
func_y = 0.59 * np.exp(-1.1 * func_x**2)

plt.clf()
plt.plot(bins_x, bins_y, 'x', ms=4)#, color='tab:red')
plt.plot(func_x, func_y, lw=.75, color='tab:gray')
plt.savefig(list_filename('plot/5/bins'), dpi=dpi)


## Fig. 6
epsilon = 0.5

data_b = np.genfromtxt('harmonic_b.csv', delimiter=',')
if len(data_b.shape) == 1:
    data_b = np.expand_dims(data_b, 0)
rows, cols = data_b.shape

correlation_x = np.array(tuple(m*epsilon for m in range(cols)))
correlation_y = np.array(tuple(correlation_function_periodic(data_b, m, epsilon) for m in range(cols)))
# j = 1
# correlation_y = np.array(tuple(correlation_function_alt(data_b, m, epsilon, 1) for m in range(cols-j)))
# correlation_x = np.array(tuple(m*epsilon for m in range(cols-j)))

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
plt.xlim(0.0, 3.0)
# plt.clf()
ax.plot(correlation_x, correlation_y, 'x', ms=4)
plt.savefig(list_filename('plot/6/correlation'), dpi=dpi)
## TODO: find low-lying energy levels


#### Fig.7
data_anharmonic_a = np.genfromtxt('anharmonic_a.csv', delimiter=',')
data_anharmonic_b = np.genfromtxt('anharmonic_b.csv', delimiter=',')
data_anharmonic_c = np.genfromtxt('anharmonic_c.csv', delimiter=',')
data_anharmonic = (data_anharmonic_a, data_anharmonic_b, data_anharmonic_c)
if len(data_anharmonic_a.shape) > 1: print("Obacht")
f_sq_tup = (0.5, 1.0, 2.0)

N = len(data_anharmonic_a)
plt.clf()
fig, axs = plt.subplots(1, 3)

for i in range(len(axs)):
    ax = axs[i]
    data = data_anharmonic[i]

    # f = np.sqrt(f_sq_tup[i])
    f_sq = f_sq_tup[i]
    f = np.sqrt(f_sq)

    ax.vlines((-f, f), 0, N+1, linestyles='dashed', color='tab:blue')
    ax.plot(data, range(N), lw=.75, color='tab:gray')#, linestyles='dash-dotted')
    ax.plot(data, range(N), 'x', ms=ms, color='tab:red')#, linestyles='dash-dotted')
    ax.set_ylim(0.0, N+1)
    ax.set_title('$f^2 = %.1f$' % f_sq)

fig.suptitle('Fig. 7')
fig.savefig(list_filename('plot/7/anharmonic'), dpi=dpi)


#### Fig. 8
data = np.genfromtxt('anharmonic_d.csv', delimiter=',')

xlower, xupper = -3., 3.
bins_x, bins_y = bin_normalized(data, 60, xlower, xupper)

# TODO: add theoretical function

# func_x = np.linspace(xlower, xupper, 200)
# func_y = 0.59 * np.exp(-1.1 * func_x**2)

plt.clf()
plt.plot(bins_x, bins_y, 'x', ms=4)#, color='tab:red')
# plt.plot(func_x, func_y, lw=.75, color='tab:gray')
plt.savefig(list_filename('plot/8/bins'), dpi=dpi)


#### Fig. 9
epsilon = 0.25

anharmonic_correlation = []
for i in "abc":
    anharmonic_correlation.append(np.genfromtxt('anharmonic_correlation_%s.csv' % i, delimiter=','))

plt.clf()
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(0.0, 3.0)

markers = ('x', '+', '4')
for data, marker in zip(anharmonic_correlation, markers):
    rows, cols = data.shape
    correlation_x = np.array(tuple(m*epsilon for m in range(cols)))
    correlation_y = np.array(tuple(correlation_function_periodic(data, m, epsilon) for m in range(cols)))

    ax.plot(correlation_x, correlation_y, marker, ms=4)

fig.savefig(list_filename('plot/9/correlation'), dpi=dpi)


#### Fig. 10
#######
## TODO: find low-lying energy levels
#######


#### Fig. 11