"""
=====================================================================
Channel dynamics
=====================================================================
Solving a Hodgkin-Huxley system and visualizing the channel dynamics.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import seaborn as sns
from matplotlib import gridspec

sns.set(context="paper", style='whitegrid', rc={"axes.facecolor": "0.96"})

###############################################################################
# Initialize the Hodgkin-Huxley system and set simulation parameters:

hh = nm.HodgkinHuxley()
T = 100         # Simulation time [ms]
dt = 0.01      # Time step


def stimulus(t):
    """
    External Current

    |  :param t: time
    |  :return: step up to 10 uA/cm^2 at t>100
    |           step down to 0 uA/cm^2 at t>200
    |           step up to 35 uA/cm^2 at t>300
    |           step down to 0 uA/cm^2 at t>400
    """
    return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300) - 35 * (t > 400)


def stimulus(t):
    # return np.where(10 <= t <= 15, 10, 0)
    # return np.where((t >= 5) & (t <= 10), 10, 0)
    # return 35 * (t > 10) - 35 * (t > 70)
    return np.where((t > 10) & (t < 70), 35, 0)


'''
def stimulus(t):
    return 10 if 5 <= t <= 7 else 0
'''
#stimulus = 10

###############################################################################
# The system is solved by calling the class method ``solve``, and the
# solutions can be accessed as class attributes:
hh.solve(stimulus, T, dt)

t = hh.t
V = hh.V
n = hh.n
m = hh.m
h = hh.h

#plt.plot(V, hh.alpha_n(V))

#plt.plot(t, V)
# plt.show()


fig = plt.figure(figsize=(7, 5), tight_layout=True, dpi=100)
gs = gridspec.GridSpec(4, 1, height_ratios=[4, 4, 4, 1.5])

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
plt.plot(t, hh.I_K(V, n), label='$I_K$')
plt.plot(t, hh.I_Na(V, m, h), label='$I_{Na}$')
plt.plot(t, hh.I_L(V), label='$I_L$')
plt.ylabel('Current')
plt.legend(loc='upper right')
ax.set_xticks([])

ax = plt.subplot(gs[3])
plt.plot(t, stimulus(t), 'k')
plt.xlabel('Time (ms)')
plt.ylabel('Input (nA)')
#ax.set_xticks([0, 10, 25, 40, np.max(t)])
ax.set_yticks([0, np.max(stimulus(t))])

plt.show()
