#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import neuromodels as nm
import neuromodels.stimuli as stim
import numpy as np
import seaborn as sns
from matplotlib import gridspec

#from neuromodels.stimuli import constant

sns.set()
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('text', usetex=True)

# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# simulation parameters
T = 50.
dt = 0.025
I_amp = 1
r_soma = 40

# input stimulus
stimulus = stim.constant(I_amp=I_amp,
                         T=T,
                         dt=dt,
                         t_stim_on=10,
                         t_stim_off=40,
                         r_soma=r_soma)
# print(stimulus["info"])
I = stimulus["I"]
I_stim = stimulus["I_stim"]

# HH simulation
hh = nm.HodgkinHuxley()
hh.solve(I, T, dt, method='Radau')
t = hh.t
V = hh.V
n = hh.n
m = hh.m
h = hh.h

# plot voltage trace
fig = plt.figure(figsize=(8, 7), tight_layout=True, dpi=120)
gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 1])
ax = plt.subplot(gs[0])
plt.plot(t, V, lw=2)
plt.ylabel('Voltage (mV)')
ax.set_xticks([])
ax.set_yticks([-80, -55, -20, 10, 40])

ax = plt.subplot(gs[1])
plt.plot(t, n, '-.', lw=1.5, label='$n$')
plt.plot(t, m, "--", lw=1.5, label='$m$')
plt.plot(t, h, ls=':', lw=1.5, label='$h$')
plt.legend(loc='upper right')
plt.ylabel("State")
ax.set_xticks([])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

ax = plt.subplot(gs[2])
plt.plot(t, I_stim, 'k', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus (nA)')

ax.set_xticks([0, 10, 25, 40, np.max(t)])
ax.set_yticks([0, np.max(I_stim)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

fig.suptitle("Hodgkin-Huxley Model")
plt.show()
