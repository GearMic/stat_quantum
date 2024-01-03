#### for plotting

import numpy as np
import matplotlib.pyplot as plt
import os.path

dpi = 200

# ax = plt.gca()

measurements = np.genfromtxt('out.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
rows, cols = measurements.shape
print(measurements.shape)
t = range(cols)

for row in range(3):
    plt.plot(t, measurements[row])

# plt.ylim(-3, 3)

i=0
while True:
    filename = "plot/plot%i.png" % i
    if os.path.isfile(filename):
        i += 1
        continue

    plt.savefig(filename, dpi=dpi)
    break


filename_bins = "plot/bins.png"
bins = np.genfromtxt('bins.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
bins_x, bins_y = bins[0], bins[1]
plt.clf()
plt.scatter(bins_x, bins_y)
plt.savefig(filename_bins, dpi=dpi)