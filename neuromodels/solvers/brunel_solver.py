#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import elephant.statistics as es
import matplotlib.pyplot as plt
import nest
#import neuromodels.raster_plot as raster_plot
import numpy as np
import quantities as pq
import seaborn as sns
from matplotlib import gridspec
from neo.core import SpikeTrain
from nest import raster_plot, visualization, voltage_trace

sns.set(context="paper", style='whitegrid', rc={"axes.facecolor": "0.96"})

#from neuromodels import raster_plot

nest.set_verbosity(level="M_FATAL")

# kw: set_state="AI"

'''
Based on the implementations:

http://arken.nmbu.no/~plesser/publications/Gewa_2012_533_preprint.pdf
https://github.com/pnquanganh/opencl-nest/blob/9c1c3c52e85f2244ca483bebf3d2bb0a2049c1de/nest-opencl/doc/nest_by_example/NEST_by_Example.ipynb

https://github.com/INM-6/hybridLFPy/blob/master/examples/brunel_alpha_nest.py
https://github.com/simetenn/uncertainpy/blob/master/examples/brunel/brunel.py
https://github.com/CINPLA/Skaar_et_al_2020_PLoS_Comput_Biol/blob/master/figure_scripts/figure_6.py


See also:

https://nest-simulator.readthedocs.io/en/v3.0/ref_material/pynest_apis.html
https://www.nest-simulator.org/py_sample/sensitivity_to_perturbation/
https://www.nest-simulator.org/py_sample/brunel_alpha_nest/
https://nest-simulator.readthedocs.io/en/stable/models/index.html
'''


class NetworkNotSimulated(Exception):
    """Failed attempt at accessing solutions.

    A call to simulate the network must be
    carried out before the solution properties
    can be used.
    """
    pass


'''
def brunel_network(eta=2, g=2, delay=1.5, J=0.1):

    # Network parameters
    N_rec = 20             # Record from 20 neurons
    simulation_end = 1000  # Simulation time

    tau_m = 20.0           # Time constant of membrane potential in ms
    V_th = 20.0
    N_E = 10000            # Number of excitatory neurons
    N_I = 2500             # Number of inhibitory neurons
    N_neurons = N_E + N_I  # Number of neurons in total
    C_E = int(N_E/10)      # Number of excitatory synapses per neuron
    C_I = int(N_I/10)      # Number of inhibitory synapses per neuron
    J_I = -g*J             # Amplitude of inhibitory postsynaptic current
    cutoff = 100           # Cutoff to avoid transient effects, in ms



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

###
order=100,
epsilon=0.1,     # OK
eta=2.0,
g=5.0,
J=0.1,           # OK
C_m=1.,          # OK
V_rest=0.,       # OK
V_th=20.,        # OK
V_reset=10.,     # OK
tau_m=20.,       # OK
tau_rp=2.,       # OK
D=1.5            # OK
'''


class BrunelNetworkSolver:
    """
    neuronal network composed of excitatory and inhibitory spiking neurons

    Implementation of the sparsely connected recurrent network described by
    Brunel (2000).

    Default parameters are chosen for the asynchronous irregular (AI) state.

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

    References
    ----------
    Brunel N, Dynamics of Sparsely Connected Networks of Excitatory and
    Inhibitory Spiking Neurons, Journal of Computational Neuroscience 8,
    183-208 (2000).
    """

    def __init__(
            self,
            order=100,
            epsilon=0.1,
            eta=2.0,
            g=5.0,
            J=0.1,
            C_m=1.,  # 250., 1.
            V_rest=0.,
            V_th=20.,
            V_reset=10.,
            tau_m=20.,
            tau_rp=2.,
            D=1.5
    ):
        """Initialize the simulation, set up data directory



        dt = 0.1    # the resolution in ms
        simtime = 1000.0  # Simulation time in ms
        delay = 1.5    # synaptic delay in ms

        '''
        Definition of the parameters crucial for asynchronous irregular firing
        of the neurons.
        '''

        g = 6.0  # ratio inhibitory weight/excitatory weight
        eta = 2.0  # external rate relative to threshold rate

        """

        # Check input

        # Network parameters
        # -------------------------------------------------------------------
        self._NE = 4 * int(order)        # number of excitatory neurons
        self._NI = 1 * int(order)        # number of inhibitory neurons
        self._N_neurons = self._NE + self._NI   # total number of neurons

        self._epsilon = epsilon           # connection probability

        # number of excitatory synapses per neuron
        self._CE = int(self._epsilon * self._NE)
        # number of inhibitory synapses per neuron
        self._CI = int(self._epsilon * self._NI)
        # total number of synapses per neuron
        self._C = self._CE + self._CI
        # -------------------------------------------------------------------

        # Neuron parameters
        # -------------------------------------------------------------------
        self._eta = eta         # background rate
        self._g = g             # relative strength of inhibitory synapses
        self._J = J             # absolute excitatory strength
        self._V_th = V_th       # firing threshold
        self._tau_m = tau_m     # membrane time constant
        self._D = D             # synaptic delay
        # self._C_m = C_m
        # self._V_rest = V_rest
        # self._V_reset = V_reset
        # self._tau_rp = tau_rp
        # -------------------------------------------------------------------

        self._neuron_params = {'C_m': C_m,
                               'tau_m': self._tau_m,
                               't_ref': tau_rp,
                               'E_L': V_rest,
                               'V_th': self._V_th,
                               'V_reset': V_reset}

        # Flags
        self._is_simulated = False
        self._is_calibrated = False
        self._is_built = False
        self._is_connected = False

    def _calibrate(self):
        """Compute dependent variables"""

        # Excitatory PSP amplitude
        self._J_ex = self._J

        # Inhibitory PSP amplitude
        self._J_in = - self._g * self._J

        # Threshold rate; the external rate needed for a neuron to reach
        # threshold in absence of feedback
        self._nu_th = self._V_th / (self._J * self._CE * self._tau_m)

        # External firing rate; firing rate of a neuron in the external
        # population
        self._nu_ext = self._eta * self._nu_th

        # Population rate of the whole external population. With C_E neurons,
        # the population rate is simply the product nu_ext*C_E. The factor
        # 1000.0 in the product changes the units from spikes per ms to
        # spikes per second.

        # Population rate of the whole external population; the product of the
        # Poisson generator rate and the in-degree C_E. The factor 1000.0
        # in the product changes the units from spikes per ms to
        # spikes per second, i.e. the rate is converted to Hz.
        self._p_rate = 1000.0 * self._nu_ext * self._CE

    def _build_network(self):
        """Create and connect network elements.

        NEST recommends that all elements in the network, i.e., neurons,
        stimulating devices and recording devices, should be created before
        creating any connections between them.
        """
        # Creating network nodes:
        #
        # The command `Create` is used to produce all node types.
        #
        # The first argument is a string denoting the type of node we want to
        # create.
        #
        # The second parameter of `Create` is an integer representing the
        # number of nodes of that type we want to create.
        #
        # The third parameter is either a dictionary or a list of dictionaries,
        # specifying the parameter settings for the created nodes. If only one
        # dictionary is given, the same parameters are used for all created
        # nodes. If an array of dictionaries is given, they are used in order
        # and their number must match the number of created nodes.

        # Set parameters for neurons
        nest.SetDefaults("iaf_psc_delta", self._neuron_params)
        # nest.SetDefaults("poisson_generator", {"rate": self._p_rate})

        # Create local excitatory neuron population
        self._nodes_ex = nest.Create("iaf_psc_delta", self._NE)
        # Create local inhibitory neuron population
        self._nodes_in = nest.Create("iaf_psc_delta", self._NI)

        # Distribute membrane potentials to random values between zero and
        # threshold
        nest.SetStatus(self._nodes_ex, "V_m",
                       np.random.rand(len(self._nodes_ex)) * self._V_th)
        nest.SetStatus(self._nodes_in, "V_m",
                       np.random.rand(len(self._nodes_in)) * self._V_th)

        self._Vm_ini_ex = np.array(nest.GetStatus(self._nodes_ex, 'V_m'))
        self._Vm_ini_in = np.array(nest.GetStatus(self._nodes_in, 'V_m'))
        # Create external population. The 'poisson_generator' device produces
        # a spike train governed by a Poisson process at a given rate. If a
        # Poisson generator is connected to N targets, it generates N i.i.d.
        # spike trains. Thus, we only need one generator to model an entire
        # population of randomly firing neurons.
        noise = nest.Create("poisson_generator", 1, {"rate": self._p_rate})

        # Create spike recorders to observe how the neurons in the recurrent
        # network respond to the random spikes from the external population.
        #
        # We create one recorder for each neuron population (excitatory and
        # inhibitory).
        #
        # By default, spike recorders record to memory but not to file. In
        # order to override this default behaviour to also record to file,
        # set the function parameter 'to_file' to True. The default file
        # names are automatically generated from the device type and its
        # global ID. We use the third argument of `Create` to give each spike
        # recorder a 'label', which will be part of the name of the output
        # file written by the recorder. Since two devices are created, we
        # supply a list of dictionaries.
        # nest.SetDefaults('spike_recorder', {'to_file': self._to_file})
        '''
        self._spikes = nest.Create("spike_recorder", 2,
                                   [{"label": 'brunel-py-ex'},
                                    {"label": 'brunel-py-in'}])
        '''
        self._spikes = nest.Create("spike_detector", 2,
                                   [{"label": 'brunel-py-ex'},
                                    {"label": 'brunel-py-in'}])
        self._espikes = self._spikes[:1]
        self._ispikes = self._spikes[1:]

        '''
        # Configuration of the spike recorders that record excitatory and
        # inhibitory spikes. `SetStatus` expects a list of node handles and
        # a list of parameter dictionaries. Setting the variable "to_file"
        # to True ensures that the spikes will be recorded in a .gdf file
        # starting with the string assigned to label. Setting "withtime" and
        # "withgid" to True ensures that each spike is saved to file by
        # stating the gid of the spiking neuron and the spike time in one line.
        nest.SetStatus(espikes, [{
        "label": os.path.join(destination, "brunel-py-ex"),
        "record_to": 'ascii',
        "withtime": True,
        "withgid": True,
        }])

        nest.SetStatus(ispikes, [{
        "label": os.path.join(destination, "brunel-py-in"),
        "record_to": 'ascii',
        "withtime": True,
        "withgid": True,
        }])
        '''

        # Configure synapse using `CopyModel`, which expects the model name
        # of a pre-defined synapse, the name of the customary synapse and
        # an optional parameter dictionary
        nest.CopyModel("static_synapse", "excitatory", {
                       "weight": self._J_ex, "delay": self._D})
        nest.CopyModel("static_synapse", "inhibitory", {
                       "weight": self._J_in, "delay": self._D})

        # Connecting network nodes:
        #
        # The function `Connect` expects four arguments: a list of source nodes,
        # a list of target nodes, a connection rule, and a synapse
        # specification (syn_spec).
        #
        # Some connection rules, in particular 'one_to_one' and 'all_to_all'
        # require no parameters and can be specified as strings. All other
        # connection rules must be specified as a dictionary, which at least
        # must contain the key 'rule' specifying a connection rule.
        #
        # The synaptic properties are inserted via syn_spec which expects a
        # dictionary when defining multiple variables or a string when simply
        # using a pre-defined synapse.

        # Connect 'external population' Poisson generator to the local
        # excitatory and inhibitory neurons using the excitatory synapse.
        # Since the Poisson generator is connected to all neurons in the local
        # populations, the default rule, 'all_to_all', of `Connect` is used.
        nest.Connect(noise, self._nodes_ex,
                     'all_to_all', syn_spec='excitatory')
        nest.Connect(noise, self._nodes_in,
                     'all_to_all', syn_spec='excitatory')

        # Connect subset of the nodes of the excitatory and inhibitory
        # populations to the associated spike recorder using excitatory
        # synapses.
        nest.Connect(self._nodes_ex[:self._N_rec], self._espikes,
                     'all_to_all', syn_spec='excitatory')
        nest.Connect(self._nodes_in[:self._N_rec], self._ispikes,
                     'all_to_all', syn_spec='excitatory')

        # Connect the excitatory population to all neurons using the
        # pre-defined excitatory synapse. Beforehand, the connection parameters
        # are defined in a dictionary. Here, we use the connection rule
        # 'fixed_indegree', which requires the definition of the indegree.
        # Since the synapse specification is reduced to assigning the
        # pre-defined excitatory synapse it suffices to insert a string.
        conn_params_ex = {'rule': 'fixed_indegree', 'indegree': self._CE}
        nest.Connect(self._nodes_ex, self._nodes_ex + self._nodes_in,
                     conn_params_ex, "excitatory")

        # Connect the inhibitory population to all neurons using the
        # pre-defined inhibitory synapse. The connection parameter as well as
        # the synapse paramtere are defined analogously to the connection from
        # the excitatory population defined above.
        conn_params_in = {'rule': 'fixed_indegree', 'indegree': self._CI}
        nest.Connect(self._nodes_in, self._nodes_ex + self._nodes_in,
                     conn_params_in, "inhibitory")

    def simulate(
        self,
        T=1000,
        dt=0.1,
        cutoff=0,
        N_rec=100,
        threads=1,
        print_time=False
    ):
        """Simulate the model


        Parameters
        ----------
        T : {int, float}, optional
            Simulation time in ms
        dt : float, optional
            Time resolution in ms
        cutoff : int, optional
            Cutoff to avoid transient effects. In unit ms.
        N_rec : int, optional
            Number of neurons to record
        """

        self._T = T
        self._N_rec = N_rec
        self._cutoff = cutoff

        # Start a new NEST session
        nest.ResetKernel()

        # Configure kernel
        # nest.SetKernelStatus({"grng_seed": 10})

        '''
        Configuration of the simulation kernel by the previously defined time
        resolution used in the simulation. Setting "print_time" to True prints
        the already processed simulation time as well as its percentage of the
        total simulation time.
        '''

        nest.SetKernelStatus({"resolution": dt,
                              "print_time": print_time,
                              "local_num_threads": threads})

        # calibrate/compute network parameters
        self._calibrate()

        # build network
        start_time_build = time.time()
        self._build_network()
        self._build_time = time.time() - start_time_build

        # simulate network
        start_time_simulate = time.time()
        nest.Simulate(self._T)
        self._simulation_time = time.time() - start_time_simulate

        '''
        Reading out the total number of spikes received from the spike
        detector connected to the excitatory population and the inhibitory
        population.
        '''

        self._events_ex = nest.GetStatus(self._espikes, "n_events")[0]
        self._events_in = nest.GetStatus(self._ispikes, "n_events")[0]

        ex_events = nest.GetStatus(self._espikes, 'events')[0]
        in_events = nest.GetStatus(self._ispikes, 'events')[0]
        ex_spikes = np.stack((ex_events['senders'], ex_events['times'])).T
        in_spikes = np.stack((in_events['senders'], in_events['times'])).T

    @property
    def num_synapses(self):
        '''
        Reading out the number of connections established using the excitatory
        and inhibitory synapse model. The numbers are summed up resulting in
        the total number of synapses.
        '''
        num_synapses = nest.GetDefaults("excitatory")["num_connections"] +\
            nest.GetDefaults("inhibitory")["num_connections"]
        return num_synapses

    @property
    def build_time_wclock(self):
        return self._build_time

    @property
    def sim_time_wclock(self):
        return self._simulation_time

    @property
    def rate_ex(self):
        '''
        Calculation of the mean firing rate of the excitatory neurons by
        dividing the total number of recorded spikes by the number of
        neurons recorded from and the simulation time. The multiplication
        by 1000.0 converts the unit 1/ms to 1/s=Hz.
        '''
        try:
            rate_ex = self._events_ex / self._T * 1000.0 / self._N_rec
            return rate_ex
        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    @property
    def rate_in(self):
        '''
        Calculation of the average firing rate of the inhibitory neurons
        by dividing the total number of recorded spikes by the number of
        neurons recorded from and the simulation time. The multiplication
        by 1000.0 converts the unit 1/ms to 1/s=Hz.
        '''
        try:
            rate_in = self._events_in / self._T * 1000.0 / self._N_rec
            return rate_in
        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    def plot_raster(self, neuron="ex", hist=True, **kwargs):
        """
        Raster plot

        **kwargs:
            Are passed to Nest's `raster_plot`:
                hist : bool, optional
                    Display histogram
                hist_binwidth : float, optional
                    Width of histogram bins
                grayscale : bool, optional
                    Plot in grayscale
                title : str, optional
                    Plot title
                xlabel : str, optional
                    Label for x-axis
        """
        if not isinstance(neuron, str):
            msg = ("'neuron' must be passed as str, either 'ex'"
                   " (excitatory) or 'in' (inhibitory).")
            raise TypeError(msg)
        if neuron not in ['ex', 'in']:
            msg = ("'neuron' must be set as either 'ex' (excitatory) or"
                   " 'in' (inhibitory).")
            raise ValueError(msg)

        try:
            if neuron == "ex":
                spikes = self._espikes
            elif neuron == "in":
                spikes = self._ispikes
            raster_plot.from_device(spikes, hist=hist, **kwargs)
            plt.show()
        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    def plot_vm_init(self):
        """
        Plot initial membrane potential distribution
        """
        plt.scatter(np.arange(self._NE), self._Vm_ini_ex, color="C0",
                    label="Excitatory")
        plt.scatter(np.arange(self._NI), self._Vm_ini_in, color="C1",
                    label="Inhibitory")
        plt.xlabel('Neuron')
        plt.ylabel('Initial membrane potential $V_m$ [mV]')
        plt.legend()
        plt.show()

    def print_network(self):
        nest.PrintNetwork()

    def plot_network(self):
        visualization.plot_network(self._nodes_ex, 'nodes_ex.png')
        # plt.show()

    def spiketrains2(self, neuron="ex"):

        # call _check_neuron
        if neuron == "ex":
            spiketrains = self.spiketrains_ex
        elif neuron == "in":
            spiketrains = self.spiketrains_in

        neo_spiketrains = []
        for spiketrain in spiketrains:
            neo_spiketrain = SpikeTrain(spiketrain,
                                        t_stop=self.t_stop,
                                        units=pq.ms)
            neo_spiketrains.append(neo_spiketrain)

        return neo_spiketrains

    def spiketrains(self, n_type="exc"):
        try:
            # check and get neuron population
            self._check_n_type(n_type)
            if n_type == "exc":
                events = nest.GetStatus(self._espikes, 'events')[0]
                nodes = self._nodes_ex
            elif n_type == "inh":
                events = nest.GetStatus(self._ispikes, 'events')[0]
                nodes = self._nodes_in

            # List of spike trains
            neo_spiketrains = []
            for sender in nodes[:self._N_rec]:

                st = events['times'][events['senders'] == sender]
                #st = st[st > self._cutoff] - self._cutoff
                id = events['senders'][events['senders'] == sender][0]
                neo_spiketrain = SpikeTrain(st,
                                            t_stop=self.t_stop,
                                            units=pq.ms,
                                            n_type=n_type,
                                            unitID=id)
                neo_spiketrains.append(neo_spiketrain)

            return neo_spiketrains

        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    @property
    def t_stop(self):
        return self._T - self._cutoff

    @property
    def spiketrains_ex(self):
        """ Excitatory spike trains
        """
        try:
            events_ex = nest.GetStatus(self._espikes, 'events')[0]
            spiketrains_ex = []
            for sender in self._nodes_ex[:self._N_rec]:
                spiketrain = events_ex['times'][events_ex['senders'] == sender]
                spiketrain = spiketrain[spiketrain >
                                        self._cutoff] - self._cutoff
                spiketrains_ex.append(spiketrain)

            return spiketrains_ex

        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    @property
    def spiketrains_in(self):
        """ Inhibitory spike trains
        """
        try:
            events_in = nest.GetStatus(self._ispikes, 'events')[0]
            spiketrains_in = []
            for sender in self._nodes_in[:self._N_rec]:
                spiketrain = events_in['times'][events_in['senders'] == sender]
                spiketrain = spiketrain[spiketrain >
                                        self._cutoff] - self._cutoff
                spiketrains_in.append(spiketrain)

            return spiketrains_in

        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    def summary(self):
        try:
            num_excitatory = int(self._CE * self._N_neurons) + self._N_neurons
            num_inhibitory = int(self._CI * self._N_neurons)
            print("-" * 40)
            print("Brunel network simulation")
            print("-" * 40)
            print(f"Build time (wall clock)      : {self._build_time:.3f} sec")
            print(
                f"Simulation time (wall clock) : {self._simulation_time:.3f} sec")
            print(f"Number of neurons            : {self._N_neurons}")
            print(f"Number of synapses           : {self.num_synapses}")
            print(f"        Excitatory           : {num_excitatory}")
            print(f"        Inhibitory           : {num_inhibitory}")
            print(f"Excitatory rate              : {self.rate_ex:.2f} Hz")
            print(f"Inhibitory rate              : {self.rate_in:.2f} Hz")
            print("-" * 40)
        except AttributeError:
            msg = ("Missing call to simulate. No solution exists.")
            raise NetworkNotSimulated(msg)

    # Check user input
    def _check_type_int_float(self, parameter, name):
        if not isinstance(parameter, (int, float)):
            msg = (f"{name} must be set as an int or float.")
            raise TypeError(msg)

    def _check_n_type(self, n_type):
        """Check whether neuron type is provided as 'exc' or 'inh'."""
        #
        if not isinstance(n_type, str):
            msg = ("'n_type' must be passed as str, either 'exc'"
                   " (excitatory) or 'inh' (inhibitory).")
            raise TypeError(msg)
        if n_type not in ['exc', 'inh']:
            msg = ("'n_type' must be set as either 'exc' (excitatory) or"
                   " 'in' (inhibitory).")
            raise ValueError(msg)

    # Get and set model parameters
    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, eta):
        #self._check_type_int_float(eta, 'eta')
        self._eta = eta

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, g):
        #self._check_type_int_float(g, 'g')
        self._g = g

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        #self._check_type_int_float(J, 'J')
        self._J = J


if __name__ == "__main__":
    bnet = BrunelNetwork(order=2500)
    bnet.g = 4.5
    bnet.eta = 2.0
    bnet.J = 0.35
    bnet.simulate(threads=8, print_time=True)
    bnet.summary()
    bnet.plot_raster()
    spiketrains = bnet.spiketrains(n_type="exc")
    bnet.summary()

    '''
    average_firing_rates = []

    print(spiketrains[0])

    for spiketrain in spiketrains:
        average_firing_rate = es.mean_firing_rate(spiketrain)
        average_firing_rate.units = pq.Hz
        # print(average_firing_rate)
        average_firing_rates.append(average_firing_rate.magnitude)

    print(f"Elephant average firing rate: {np.mean(average_firing_rates)}")
    '''

    #plt.plot(times, average_firing_rates, 'o')
    # plt.show()
    #frate = mean_firing_rate(train)
    # print(frate)
    # print(frate.magnitude)
    # bnet.summary()
    # bnet.plot_vm_init()
    #spiketrains_ex = bnet.spiketrains_ex

    # bnet.plot_raster(title="yo")
    # bnet.plot_raster_in()
    # bnet.plot_raster()
    # bnet.plot_network()
    # bnet.plot_nodes()
    # bnet.plot_raster_in()
    # print("DONE 2")
    # bnet.simulate()
    # print("h")
