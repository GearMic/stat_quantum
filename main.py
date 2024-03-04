from montecarlo import *

import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy.optimize as optimize

# helper constants
dpi = 200
ms = 4


# plot action over time
fig, ax = plt.subplots()
data = np.genfromtxt('action.csv', delimiter=',')
plt.plot(range(len(data)), data) 
plt.savefig('plot/0_plot.png', dpi=dpi)


# Fig.4
data = np.genfromtxt('harmonic_a.csv', delimiter=',')
rows, cols = data.shape
t = range(cols)

plt.clf()
for row in range(3):
    plt.plot(t, data[row])

plt.savefig('plot/4_plot.png', dpi=dpi)


# Fig. 5
data = data[:, :-1]

xlower, xupper = -3., 3.
bins_x, bins_y = bin_normalized(data, 31, xlower, xupper)

func_x = np.linspace(xlower, xupper, 200)
func_y = 0.59 * np.exp(-1.1 * func_x**2) # theoretical distribution from the paper

plt.clf()
plt.plot(bins_x, bins_y, '.', ms=1)#, color='tab:red')
plt.plot(func_x, func_y, lw=.75, color='tab:gray')
plt.savefig('plot/5_bins.png', dpi=dpi)


## Fig. 6
a = 0.5
data = np.genfromtxt('harmonic_b.csv', delimiter=',')

correlation_x, correlation_y, correlation_std = correlation_function(data, a)
# theoretical line from paper
theory_x = np.array((0.0, 2.5))
theory_y = np.array((0.45, 0.004))

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
plt.xlim(0.0, 2.5)
ax.plot(theory_x, theory_y, color='tab:gray')
# ax.plot(correlation_x, correlation_y, 'x', ms=4) # no errorbars
ax.errorbar(correlation_x, correlation_y, correlation_std) # errorbars
plt.savefig('plot/6_correlation.png', dpi=dpi)

fig, ax = plt.subplots()
ax.plot(correlation_x, correlation_y, 'x', ms=4)
plt.savefig('plot/6_correlation_linear.png', dpi=dpi)

# find Energy eigenvalues
# prepare exponential fit
def exp_fn(x: float, a: float, b: float):
    return a * np.exp(b*x)
initial_guess = (0.5, -0.1)

# fit for E0
popt, pcov = optimize.curve_fit(exp_fn, correlation_x, correlation_y, initial_guess, correlation_std)
a, b = popt
fit_y = np.array(tuple(exp_fn(x, a, b) for x in correlation_x))
E = np.abs(b)
E_err = np.sqrt(np.diag(pcov))[1]
print("E = %f" % E)


## Fig.7
letters = "abc"
data_tup = tuple(np.genfromtxt('anharmonic_%s.csv' % letter, delimiter=',') for letter in letters)
f_sq_tup = (0.5, 1.0, 2.0)

N = len(data_tup[0])
plt.clf()
fig, axs = plt.subplots(1, len(data_tup))

for i in range(len(data_tup)):
    ax = axs[i]
    data = data_tup[i]
    f_sq = f_sq_tup[i]
    f = np.sqrt(f_sq)

    ax.vlines((-f, f), 0, N+1, linestyles='dashed', color='tab:blue')
    ax.plot(data, range(N), lw=.75, color='tab:gray')#, linestyles='dash-dotted')
    ax.plot(data, range(N), 'x', ms=ms, color='tab:red')#, linestyles='dash-dotted')
    ax.set_ylim(0.0, N+1)
    ax.set_title('$f^2 = %.1f$' % f_sq)

fig.suptitle('Fig. 7')
fig.savefig('plot/7_anharmonic.png', dpi=dpi)


## Fig. 8
data = np.genfromtxt('anharmonic_d.csv', delimiter=',')

x_bound = 2.5
xlower, xupper = -x_bound, x_bound
bins_x, bins_y = bin_normalized(data, 60, xlower, xupper)

# TODO: add theoretical distribution

plt.clf()
plt.plot(bins_x, bins_y, 'x', ms=ms)
plt.savefig('plot/8_bins.png', dpi=dpi)


## Fig. 9
a = 0.25

letters = "abcd"
data_tup = tuple(np.genfromtxt('anharmonic_correlation_%s.csv' % letter, delimiter=',') for letter in letters)

plt.clf()
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.yaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(0.0, 5.0)

theory_x = np.array((0.0, 5.0))
theory_y = np.array((1.45, 0.2))
ax.plot(theory_x, theory_y, color='tab:gray')

markers = ('x', '+', '4', '*')
for data, marker, letter in zip(data_tup, markers, letters):
    correlation_x, correlation_y, correlation_std = correlation_function(data, a)

    ax.plot(correlation_x, correlation_y, marker, ms=4, label=letter)

ax.legend()
fig.savefig('plot/9_correlation.png', dpi=dpi)


## Fig. 10
a = 0.25
f_sq = (0.0, 0.5, 1.0, 1.5, 2.0)
E = np.zeros_like(f_sq)
def exp_fn(x: float, a: float, b: float):
    return a * np.exp(b*x)

initial_guess = (0.5, -0.5)
for i in range(len(f_sq)):
    data = np.genfromtxt('anharmonic_energy%i.csv' % i, delimiter=',')
    correlation_x, correlation_y, correlation_std = correlation_function(data, a)

    popt, pcov = optimize.curve_fit(exp_fn, correlation_x, correlation_y, initial_guess, correlation_std)
    E[i] = np.abs(popt[1])

plt.clf()
fig, ax = plt.subplots()
ax.plot(f_sq, E)
fig.savefig('plot/10_energy.png', dpi=dpi)