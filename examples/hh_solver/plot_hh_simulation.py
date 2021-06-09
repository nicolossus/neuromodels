"""
=====================================================================
The Hodgkin-Huxley Model Solver
=====================================================================
Solving a Hodgkin-Huxley system and visualizing the simulation results.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import seaborn as sns
from matplotlib import gridspec

sns.set(context="paper", style='whitegrid', rc={"axes.facecolor": "0.96"})

###############################################################################
# Initialize the Hodgkin-Huxley system; model parameters can either be
# set in the constructor or accessed as class attributes:

hh = nm.HodgkinHuxley(V_rest=-65)
hh.gbar_K = 36

###############################################################################
# The simulation parameters needed are the simulation time ``T``, the time
# step ``dt``, and the input ``stimulus``:

T = 50.     # Simulation time [ms]
dt = 0.01   # Time step

###############################################################################
# The ``stimulus`` must be provided as either a scalar value;

stimulus = 10   # [mV]

###############################################################################
# a callable, e.g. a function, with signature ``(t)``;


def stimulus(t):
    return 10 if 10 <= t <= 40 else 0

###############################################################################
# or a ndarray with `shape=(int(T/dt)+1,)`;


def generate_stimulus(I_amp, T, dt, t_stim_on, t_stim_off):
    time = np.arange(0, T + dt, dt)
    I_stim = np.zeros_like(time)
    stim_on_ind = int(np.round(t_stim_on / dt))
    stim_off_ind = int(np.round(t_stim_off / dt))
    I_stim[stim_on_ind:stim_off_ind] = I_amp
    return I_stim


I_amp = 10          # Input stimulus amplitude [mV]
t_stim_on = 10      # Time when stimulus is turned on [ms]
t_stim_off = 40     # Time when stimulus is turned off [ms]
stimulus = generate_stimulus(I_amp, T, dt, t_stim_on, t_stim_off)

###############################################################################
# The system is solved by calling the class method ``solve``:

hh.solve(stimulus, T, dt)

###############################################################################
# The solutions can be accessed as class attributes:

t = hh.t
V = hh.V
n = hh.n
m = hh.m
h = hh.h

###############################################################################
# The simulation results can then be plotted:

fig = plt.figure(figsize=(7, 5), tight_layout=True, dpi=300)
gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 1.5])

ax = plt.subplot(gs[0])
plt.plot(t, V)
plt.ylabel('Voltage (mV)')
ax.set_xticks([])
ax.set_yticks([-70, -20, 30])

ax = plt.subplot(gs[1])
plt.plot(t, n, label='$n$')
plt.plot(t, m, label='$m$')
plt.plot(t, h, label='$h$')
plt.ylabel("State")
plt.legend(loc='upper right')
ax.set_xticks([])
ax.set_yticks([0.0, 0.5, 1.0])


ax = plt.subplot(gs[2])
plt.plot(t, stimulus, 'k')
plt.xlabel('Time (ms)')
plt.ylabel('Input (nA)')
ax.set_xticks([0, 10, 25, 40, np.max(t)])
ax.set_yticks([0, np.max(stimulus)])

plt.show()
