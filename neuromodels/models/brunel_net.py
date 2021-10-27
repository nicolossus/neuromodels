#!/usr/bin/env python
# -*- coding: utf-8 -*-

import elephant
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import quantities as pq
import viziphant
from elephant.spike_train_correlation import correlation_coefficient
from neuromodels.solvers import BrunelNetworkSolver
from viziphant.rasterplot import rasterplot as vizi_rasterplot


class BrunelNet:

    def __init__(
        self,
        T=1000,
        dt=0.1,
        J=0.1,
        cutoff=0,
        N_rec=100,
        threads=1,
        print_time=False,
        n_type="exc",
        **kwargs
    ):
        """
        **kwargs
            Arbitrary keyword arguments are passed to the BrunelNetwork
            constructor.
        """
        self._bnet = BrunelNetworkSolver(J=J, **kwargs)
        self._T = T
        self._dt = dt
        self._cutoff = cutoff
        self._N_rec = N_rec
        self._threads = threads
        self._print_time = print_time
        self._n_type = n_type

    def __call__(self, eta=2.0, g=4.5):

        self._bnet.eta = eta
        self._bnet.g = g
        self._bnet.simulate(T=self._T,
                            dt=self._dt,
                            cutoff=self._cutoff,
                            N_rec=self._N_rec,
                            threads=self._threads,
                            print_time=self._print_time)
        self._spiketrains = self._bnet.spiketrains(n_type=self._n_type)
        return self._spiketrains

    '''
    def __call__(self, eta=2.0, g=4.5, J=0.1, n_type=None):
        if n_type is None:
            n_type = self._n_type
        self._bnet.eta = eta
        self._bnet.g = g
        self._bnet.J = J
        self._bnet.simulate(T=self._T,
                            dt=self._dt,
                            cutoff=self._cutoff,
                            N_rec=self._N_rec,
                            threads=self._threads,
                            print_time=self._print_time)
        self._spiketrains = self._bnet.spiketrains(n_type=n_type)
        return self._spiketrains
    '''

    def _check_type_quantity(self, parameter, name):
        if not isinstance(parameter, pq.Quantity):
            msg = (f"{name} must be set as a Quantity object.")
            raise TypeError(msg)

    def _slice_spiketrains(self, spiketrains, t_start=None, t_stop=None):
        if t_start is not None:
            self._check_type_quantity(t_start, 't_start')
        if t_stop is not None:
            self._check_type_quantity(t_stop, 't_stop')

        spiketrains_slice = []
        for spiketrain in spiketrains:
            if t_start is None:
                t_start = spiketrain.t_start
            if t_stop is None:
                t_stop = spiketrain.t_stop

            spiketrain_slice = spiketrain[np.where(
                (spiketrain > t_start) & (spiketrain < t_stop))]
            spiketrain_slice.t_start = t_start
            spiketrain_slice.t_stop = t_stop
            spiketrains_slice.append(spiketrain_slice)
        return spiketrains_slice

    def _statistics(self):
        # kreve statistics som en liste
        sum_stats = [getattr(sptstats, statistic) for statistic in statistics]

    def rasterplot(self, spiketrains=None, ax=None, t_start=None, t_stop=None, s=3, **kwargs):
        """
        ax: matplotlib.axes.Axes or None, optional
            Matplotlib axes handle. If None, new axes are created and returned.
            Default: None
        s: float or array-like, shape (n, ), optional
            The marker size in points**2
        **kwargs
            Arbitrary keyword arguments
        """
        if spiketrains is None:
            spiketrains = self._spiketrains

        if t_start is not None or t_stop is not None:
            spiketrains = self._slice_spiketrains(spiketrains,
                                                  t_start=t_start,
                                                  t_stop=t_stop)

        if ax is None:
            ax = plt.gca()

        vizi_rasterplot(spiketrains, axes=ax, marker='.', s=s, **kwargs)
        ax.set_ylabel("Neuron ID")
        ax.yaxis.set_major_locator(ticker.FixedLocator(5))
        ax.set_yticklabels([st.annotations['unitID']
                           for st in spiketrains[::5]])

    def rasterplot_rates(self, spiketrains=None, ax=None, t_start=None, t_stop=None, s=3, **kwargs):
        """
        ax: matplotlib.axes.Axes or None, optional
            Matplotlib axes handle. If None, new axes are created and returned.
            Default: None
        s: float or array-like, shape (n, ), optional
            The marker size in points**2
        **kwargs
            Arbitrary keyword arguments

         style='ticks',
         palette=None,
         context=None,  # paper, poster, talk

        # Initialize plotting canvas
        sns.set_style(style)

        if context is not None:
            sns.set_context(context)

        if palette is not None:
            sns.set_palette(palette)
        else:
            palette = sns.color_palette()

        if ax is None:
            fig, ax = plt.subplots()
            # axis must be created after sns.set() command for style to apply!
        """
        if spiketrains is None:
            spiketrains = self._spiketrains
        if t_start is not None or t_stop is not None:
            spiketrains = self._slice_spiketrains(spiketrains,
                                                  t_start=t_start,
                                                  t_stop=t_stop)

        viziphant.rasterplot.rasterplot_rates(spiketrains, **kwargs)

    def plot_time_histogram(spiketrains, bin_size=100 * pq.ms):
        histogram = elephant.statistics.time_histogram(spiketrains,
                                                       bin_size=100 * pq.ms)
        units = spiketrains[0].units
        #viziphant.statistics.plot_time_histogram(histogram, units=units)

    def plot_instantaneous_rate(self, spiketrains, sigma=40 * pq.ms, sampling_period=10 * pq.ms):
        """
        """
        sigma = 5 * pq.ms
        sampling_period = 1 * pq.ms

        kernel = elephant.kernels.GaussianKernel(sigma=sigma)
        rates = elephant.statistics.instantaneous_rate(spiketrains,
                                                       sampling_period=sampling_period,
                                                       kernel=kernel)
        viziphant.statistics.plot_instantaneous_rates_colormesh(rates)

    def summary(self):
        pass

    # Plot simulation
    def plot_voltage_trace(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.plot([1, 2, 3], [1, 2, 3], **kwargs)
        ax.xlabel("Time (ms)")
        ax.ylabel('Voltage (mV)')


if __name__ == "__main__":
    import neuromodels as nm
    import seaborn as sns
    from matplotlib import gridspec

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

    bnet_simulator = BrunelSimulator(order=500,
                                     T=1000,
                                     dt=0.1,
                                     N_rec=50,
                                     threads=8,
                                     print_time=True,
                                     n_type='exc')

    spiketrains = bnet_simulator(eta=2.0, g=4.5, J=0.35)

    # spiketrains_ex = bnet_simulator(neuron='ex')
    # spiketrains_in = bnet_simulator(neuron='in')
    # spiketrains = [spiketrains_ex, spiketrains_in]
    sts = nm.statistics.SpikeTrainStats()
    mean_firing_rate = sts.mean_firing_rate(spiketrains)
    print(f"{mean_firing_rate=}")

    t_start = 100 * pq.ms
    t_stop = 500 * pq.ms

    # print(type(spiketrains))
    # bnet_simulator.rasterplot()
    #bnet_simulator.rasterplot(t_start=t_start, t_stop=t_stop)
    bnet_simulator.rasterplot_rates(spiketrains)
    #bnet_simulator.rasterplot_rates(t_start=t_start, t_stop=t_stop)
    # bnet_simulator.plot_time_histogram(spiketrains)
    # bnet_simulator.plot_instantaneous_rate(spiketrains)
    plt.show()
