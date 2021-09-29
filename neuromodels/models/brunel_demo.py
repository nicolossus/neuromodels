
def brunel_network(eta=2, g=2, delay=1.5, J=0.1):
    """
    A sparsely connected recurrent network (Brunel).
    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).
    Parameters
    ----------
    eta : {int, float}, optional
        External rate relative to threshold rate. Default is 2.
    g : {int, float}, optional
        Ratio inhibitory weight/excitatory weight. Default is 5.
    delay : {int, float}, optional
        Synaptic delay in ms. Default is 1.5.
    J : {int, float}, optional
        Amplitude of excitatory postsynaptic current. Default is 0.1
    Notes
    -----
    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).
    """
    # Network parameters
    N_rec = 20             # Record from 20 neurons
    simulation_end = 1000  # Simulation time

    tau_m = 20.0           # Time constant of membrane potential in ms
    V_th = 20.0
    N_E = 10000            # Number of excitatory neurons
    N_I = 2500             # Number of inhibitory neurons
    N_neurons = N_E + N_I  # Number of neurons in total
    C_E = int(N_E / 10)      # Number of excitatory synapses per neuron
    C_I = int(N_I / 10)      # Number of inhibitory synapses per neuron
    J_I = -g * J             # Amplitude of inhibitory postsynaptic current
    cutoff = 100           # Cutoff to avoid transient effects, in ms

    nu_ex = eta * V_th / (J * C_E * tau_m)
    p_rate = 1000.0 * nu_ex * C_E

    nest.ResetKernel()

    # Configure kernel
    nest.SetKernelStatus({"grng_seed": 10})

    nest.SetDefaults('iaf_psc_delta',
                     {'C_m': 1.0,
                      'tau_m': tau_m,
                      't_ref': 2.0,
                      'E_L': 0.0,
                      'V_th': V_th,
                      'V_reset': 10.0})

    # Create neurons
    nodes = nest.Create('iaf_psc_delta', N_neurons)
    nodes_E = nodes[:N_E]
    nodes_I = nodes[N_E:]

    noise = nest.Create('poisson_generator', 1, {'rate': p_rate})

    spikes = nest.Create('spike_detector', 2,
                         [{'label': 'brunel-py-ex'},
                          {'label': 'brunel-py-in'}])
    spikes_E = spikes[:1]
    spikes_I = spikes[1:]

    # Connect neurons to each other
    nest.CopyModel('static_synapse_hom_w', 'excitatory',
                   {'weight': J, 'delay': delay})
    nest.Connect(nodes_E, nodes,
                 {'rule': 'fixed_indegree', 'indegree': C_E},
                 'excitatory')

    nest.CopyModel('static_synapse_hom_w', 'inhibitory',
                   {'weight': J_I, 'delay': delay})
    nest.Connect(nodes_I, nodes,
                 {'rule': 'fixed_indegree', 'indegree': C_I},
                 'inhibitory')

    # Connect poisson generator to all nodes
    nest.Connect(noise, nodes, syn_spec='excitatory')

    nest.Connect(nodes_E[:N_rec], spikes_E)
    nest.Connect(nodes_I[:N_rec], spikes_I)

    # Run the simulation
    nest.Simulate(simulation_end)

    events_E = nest.GetStatus(spikes_E, 'events')[0]
    events_I = nest.GetStatus(spikes_I, 'events')[0]

    # Excitatory spike trains
    # Makes sure the spiketrain is added even if there are no results
    # to get a regular result
    spiketrains = []
    for sender in nodes_E[:N_rec]:
        spiketrain = events_E["times"][events_E["senders"] == sender]
        spiketrain = spiketrain[spiketrain > cutoff] - cutoff
        spiketrains.append(spiketrain)

    simulation_end -= cutoff

    return simulation_end, spiketrains


"""
Skaar
"""

# Set parameters for neurons and poisson generator
nest.SetDefaults("iaf_psc_delta", neuron_params)
nest.SetDefaults("poisson_generator", {"rate": p_rate})

# Create all neurons and recorders
# local populations
nodes_ex = nest.Create("iaf_psc_delta", PSET.NE)
nodes_in = nest.Create("iaf_psc_delta", PSET.NI)

# external population
noise = nest.Create("poisson_generator")

# spike recorders
espikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")
print("first exc node: {}".format(nodes_ex[0]))
print("first inh node: {}".format(nodes_in[0]))

# Set initial membrane voltages to random values between 0 and threshold
nest.SetStatus(nodes_ex, "V_m",
               np.random.rand(len(nodes_ex)) * neuron_params["V_th"])
nest.SetStatus(nodes_in, "V_m",
               np.random.rand(len(nodes_in)) * neuron_params["V_th"])

# Spike recording parameters
nest.SetStatus(espikes, [{
    "label": os.path.join(PSET.savefolder, 'nest_output', PSET.ps_id, label + "-EX"),
    "withtime": True,
    "withgid": True,
    "to_file": PSET.save_spikes,
}])
nest.SetStatus(ispikes, [{
    "label": os.path.join(PSET.savefolder, 'nest_output', PSET.ps_id, label + "-IN"),
    "withtime": True,
    "withgid": True,
    "to_file": PSET.save_spikes, }])

# Set synaptic weights
nest.CopyModel("static_synapse", "excitatory",
               {"weight": PSET.J, 'delay': PSET.delay})
nest.CopyModel("static_synapse", "inhibitory",
               {"weight": -PSET.J * PSET.g, 'delay': PSET.delay})

# Connect 'external population' poisson generator to local neurons
nest.Connect(noise, nodes_ex, 'all_to_all', "excitatory")
nest.Connect(noise, nodes_in, 'all_to_all', "excitatory")

# Record spikes to be saved from a subset of each population
nest.Connect(nodes_ex, espikes, 'all_to_all', 'excitatory')
nest.Connect(nodes_in, ispikes, 'all_to_all', 'excitatory')


# Connect local populations
conn_params_ex = {'rule': 'fixed_indegree', 'indegree': PSET.CE}
nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex, 'excitatory')
conn_params_in = {'rule': 'fixed_indegree', 'indegree': PSET.CI}
nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in, 'inhibitory')


endbuild = time.time()

nest.Simulate(PSET.simtime)

if __name__ == "__main__":
    brunel_network()
