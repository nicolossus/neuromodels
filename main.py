import matplotlib.pyplot as plt
import neuromodels as nm

# Initialize the Hodgkin-Huxley system; model parameters can either be
# set in the constructor or accessed as class attributes:
hh = nm.HodgkinHuxley(V_rest=-70)
hh.gbar_K = 36

# The simulation parameters needed are the simulation time ``T``, the time
# step ``dt``, and the input ``stimulus``, the latter either as a
# callable or ndarray with `shape=(int(T/dt)+1,)`:
T = 50.
dt = 0.025


def stimulus(t):
    return 10 if 10 <= t <= 40 else 0


# The system is solved by calling the class method ``solve`` and the
# solutions can be accessed as class attributes:
hh.solve(stimulus, T, dt)
t = hh.t
V = hh.V

plt.plot(t, V)
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential [mV]')
plt.show()
