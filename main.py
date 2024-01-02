#### for plotting

import numpy as np
import matplotlib.pyplot as plt


measurements = np.genfromtxt('test.csv', delimiter=',')#, skip_header=5, usecols=data_usecols, converters=data_converters)
rows, cols = measurements.shape
print(measurements.shape)
t = range(cols)

for row in range(rows):
    # if row % rows == 0:
    #     plt.plot(t, measurements[row])
    #     print('test')
    plt.plot(t, measurements[row])
    print(measurements.shape)
plt.show()