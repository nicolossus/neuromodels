import matplotlib.pyplot as plt
import nest
import numpy as np

nest.set_verbosity('M_WARNING')
nest.ResetKernel()

simtime = 1000

dcfrom = 0
dcstep = 20
dcto = 2000

h = 0.1  # simulation step size in mS

neuron = nest.Create('hh_psc_alpha')
sd = nest.Create('spike_detector')

nest.SetStatus(sd, {'to_memory': False})

nest.Connect(neuron, sd, syn_spec={'weight': 1.0, 'delay': h})


n_data = int(dcto / float(dcstep))
amplitudes = np.zeros(n_data)
event_freqs = np.zeros(n_data)
for i, amp in enumerate(range(dcfrom, dcto, dcstep)):
    nest.SetStatus(neuron, {'I_e': float(amp)})
    print("Simulating with current I={} pA".format(amp))
    nest.Simulate(1000)  # one second warm-up time for equilibrium state
    nest.SetStatus(sd, {'n_events': 0})  # then reset spike counts
    nest.Simulate(simtime)  # another simulation call to record firing rate

    n_events = nest.GetStatus(sd, keys={'n_events'})[0][0]
    amplitudes[i] = amp
    event_freqs[i] = n_events / (simtime / 1000.)

plt.plot(amplitudes, event_freqs)
plt.show()
