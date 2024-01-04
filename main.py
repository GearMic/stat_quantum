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


dpi = 200

# ax = plt.gca()

measurements = np.genfromtxt('out.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
rows, cols = measurements.shape
print(measurements.shape)
t = range(cols)

for row in range(3):
    # plt.plot(t, measurements[row])
    plt.plot(t, measurements[-row-1])

# plt.ylim(-3, 3)

# i=0
# while True:
#     filename = "plot/plot%i.png" % i
#     if os.path.isfile(filename):
#         i += 1
#         continue

#     plt.savefig(filename, dpi=dpi)
#     break

plt.savefig(list_filename('plot/plot'), dpi=dpi)


# filename_bins = "plot/bins.png"
bins = np.genfromtxt('bins.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
bins_x, bins_y = bins[0], bins[1]
bin_size = bins_x[1] - bins_x[0]
bins_x += bin_size / 2 # for plotting bin value in the middle of each bin
func_x = np.linspace(bins_x[0], bins_x[-1] + bin_size, 100)
func_y = 0.59 * np.exp(-1.1 * func_x**2)

plt.clf()
plt.plot(bins_x, bins_y, 'x', ms=4)
plt.plot(func_x, func_y)
plt.savefig(list_filename('plot/bins'), dpi=dpi)