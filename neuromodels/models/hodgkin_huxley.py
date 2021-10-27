#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from neuromodels.solvers import HodgkinHuxleySolver
from neuromodels.statistics import SpikeStats


class HodgkinHuxley:
    """
    Hodgkin-Huxley simulator model
    """

    def __init__(
            self,
            stimulus,
            T,
            dt,
            y0=None,
            method='RK45',
            pdict={},
            solver_options={},
    ):
        """
        """
        self._hh = HodgkinHuxleySolver(**pdict)
        self._stim = stimulus
        self._T = T
        self._dt = dt
        self._y0 = y0
        self._method = method
        self._solver_options = solver_options

    def __call__(
        self,
        gbar_K=36.,
        gbar_Na=120.,
        noise=False,
        noise_scale=0.1,
        noise_seed=None
    ):
        self._hh.gbar_K = gbar_K
        self._hh.gbar_Na = gbar_Na

        self._hh.solve(self._stim,
                       self._T,
                       self._dt,
                       y0=self._y0,
                       method=self._method,
                       **self._solver_options
                       )

        self._V = self._hh.V
        self._t = self._hh.t

        if noise:
            rng = np.random.default_rng(noise_seed)
            self._V += rng.normal(loc=0,
                                  scale=noise_scale,
                                  size=self._V.shape)

        return self._V, self._t

    @property
    def stimulus_array(self):
        if isinstance(self._stim, (int, float)):
            stim_array = np.ones(self._t.shape) * self._stim
        elif callable(self._stim):
            stim_array = np.array([self._stim(ti) for ti in self._t])
        elif isinstance(stim, np.ndarray):
            stim_array = stim
        return stim_array

    # Model parameters
    @property
    def V_rest(self):
        return self._hh.V_rest

    @property
    def Cm(self):
        return self._hh._Cm

    @property
    def gbar_K(self):
        return self._hh._gbar_K

    @property
    def gbar_Na(self):
        return self._hh._gbar_Na

    @property
    def gbar_L(self):
        return self._hh._gbar_L

    @property
    def E_K(self):
        return self._hh._E_K

    @property
    def E_Na(self):
        return self._hh._E_Na

    @property
    def E_L(self):
        return self._hh._E_L

    @property
    def degC(self):
        return self._hh._degC

    # Solutions
    @property
    def t(self):
        return self._hh._time

    @property
    def V(self):
        return self._hh._V

    @property
    def n(self):
        return self._hh._n

    @property
    def m(self):
        return self._hh._m

    @property
    def h(self):
        return self._hh._h

    def summary_stats(self, t_stim_on, t_stim_off, threshold=0, stats=None):
        sps = SpikeStats(t_stim_on, t_stim_off, threshold, stats)
        return sps(self._V, self._t)

    def stats_df(self, t_stim_on, t_stim_off, threshold=0, sum_stats=None):
        """

        """
        stat_labels = []
        for stat in sum_stats:
            if stat == "n_spikes":
                stat = "Number of spikes"
            else:
                stat = stat[0].upper() + stat[1:]
                stat = stat.replace("_", " ")
            stat_labels.append(stat)

        sps = SpikeStats(t_stim_on, t_stim_off, threshold, sum_stats)
        stats_data = dict(zip(stat_labels, sps(self._V, self._t)))
        df = pd.DataFrame.from_dict(stats_data, orient='index').reset_index()
        df.columns = ['Statistic', 'Value']
        return df

    def plot_voltage_trace(
            self,
            with_stim=True,
            lw=1.5,
            figsize=(6, 4),
            dpi=150,
            savefig=None,
            **kwargs
    ):
        """
        """
        if with_stim:
            fig = plt.figure(figsize=figsize, tight_layout=True, dpi=dpi)
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])

            ax = plt.subplot(gs[0])
            ax.plot(self.t, self.V, lw=lw, **kwargs)
            ax.set_ylabel('Membrane Potential (mV)')
            ax.axes.xaxis.set_ticklabels([])
            # ax.set_xticks([])
            # ax.set_yticks([-70, -20, 30])

            ax = plt.subplot(gs[1])
            plt.plot(self.t, self.stimulus_array, 'k', lw=lw, **kwargs)
            plt.xlabel('Time (ms)')
            plt.ylabel(r'Stimulus ($\mu \mathrm{A/cm}^2$)')
            # ax.set_xticks([0, 10, 25, 40, np.max(t)])
            # ax.set_yticks([0, np.max(stimulus)])

            # plt.show()
        else:
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1,
                                   figsize=figsize,
                                   tight_layout=True,
                                   dpi=dpi
                                   )
            ax.plot(self.t, self.V, **kwargs)
            ax.set(xlabel="Time (ms)",
                   ylabel="Membrane Potential (mV)"
                   )

        return fig

    def plot_rates(self, rates_only=False, savefig=None):
        # fig = plt.figure(figsize=(7, 5), tight_layout=True, dpi=300)
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])

        ax = plt.subplot(gs[0])
        plt.plot(t, V)
        plt.ylabel('Membrane Potential (mV)')
        # ax.set_xticks([])
        # ax.set_yticks([-70, -20, 30])

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
        plt.ylabel('Stimulus (nA)')
        ax.set_xticks([0, 10, 25, 40, np.max(t)])
        ax.set_yticks([0, np.max(stimulus)])

    def plot_spike_statistics(
        self,
        t_stim_on,
        t_stim_off,
        threshold=0,
        figsize=(6, 4),
        dpi=150,
        savefig=None
    ):
        V = self.V
        t = self.t
        sps = SpikeStats(t_stim_on, t_stim_off, threshold)
        # plot voltage trace with features
        spikes = sps.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 3:
            msg = ("At least 3 spikes are needed in voltage trace in order "
                   "to plot spike statistics")
            raise RuntimeError(msg)

        spike_idxs = spikes["spike_idxs"]
        spike_heights = spikes["spike_heights"]
        width_lines = sps.width_lines(V, t)
        ahp_depth_idxs = sps.AHP_depth_positions(V, t)

        fig = plt.figure(figsize=figsize, tight_layout=True, dpi=dpi)
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 2])
        ax0 = plt.subplot(gs[0])

        # voltage trace
        ax0.plot(t,
                 V,
                 lw=1.5,
                 label='Voltage trace'
                 )

        # AP overshoot
        ax0.plot(t[spike_idxs],
                 V[spike_idxs],
                 "x",
                 ms=7,
                 color='black',
                 label='AP overshoot'
                 )

        # AP widths
        ax0.hlines(*width_lines,
                   color="red",
                   lw=2,
                   label='AP width'
                   )

        # AHP depths
        ax0.plot(t[ahp_depth_idxs],
                 V[ahp_depth_idxs],
                 'o',
                 ms=5,
                 color='indianred',
                 label='AHP depth'
                 )

        # latency to first spike
        ax0.hlines(self.V_rest,
                   t_stim_on,
                   t[spike_idxs[0]],
                   color='black',
                   lw=1.5,
                   ls=":"
                   )
        ax0.vlines(t[spike_idxs[0]],
                   self.V_rest,
                   spike_heights[0],
                   color='black',
                   lw=1.5,
                   ls=":",
                   label="Latency to first spike"
                   )

        # spike rate; mark spike locations
        for i in range(n_spikes):
            if i == 0:
                ax0.vlines(t[spike_idxs[i]],
                           V[spike_idxs[i]],
                           48,
                           color='darkorange',
                           ls='--',
                           label='Spike rate'
                           )
            else:
                ax0.vlines(t[spike_idxs[i]],
                           V[spike_idxs[i]],
                           48,
                           color='darkorange',
                           ls='--'
                           )
        # the marked ISIs are used to compute the accommodation index
        # ISI arrow legend
        ax0.plot([],
                 [],
                 color='g',
                 marker=r'$\longleftrightarrow$',
                 linestyle='None',
                 markersize=15,
                 label='ISIs'
                 )
        for i in range(n_spikes - 1):
            ax0.annotate('',
                         xy=(t[spike_idxs[i]], 48),
                         xycoords='data',
                         xytext=(t[spike_idxs[i + 1]], 48),
                         textcoords='data',
                         arrowprops={'arrowstyle': '<->', 'color': 'g'}
                         )

        ax0.set_ylabel('Membrane Potential (mV)')
        ax0.axes.xaxis.set_ticklabels([])

        #plt.ylim(-90, 60)
        # ax.set_xticks([])
        #ax.set_yticks([-80, -20, 40])
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles,
                   labels,
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )

        ax1 = plt.subplot(gs[1])
        ax1.plot(t, self.stimulus_array, 'k', lw=2)
        ax1.set(xlabel='Time (ms)',
                ylabel=r'Stimulus ($\mu \mathrm{A/cm}^2$)'
                )
        # ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

        fig.suptitle("Spike Statistics")
        # plt.show()
        return fig

    def plot_spike_statistics2(
        self,
        t_stim_on,
        t_stim_off,
        threshold=0,
        ax=None,
    ):
        V = self.V
        t = self.t
        sps = SpikeStats(t_stim_on, t_stim_off, threshold)
        # plot voltage trace with features
        spikes = sps.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 3:
            msg = ("At least 3 spikes are needed in voltage trace in order "
                   "to plot spike statistics")
            raise RuntimeError(msg)

        spike_idxs = spikes["spike_idxs"]
        spike_heights = spikes["spike_heights"]
        width_lines = sps.width_lines(V, t)
        ahp_depth_idxs = sps.AHP_depth_positions(V, t)

        if ax is None:
            ax = plt.gca()

        # voltage trace
        ax.plot(t,
                V,
                lw=1.5,
                label='Voltage trace'
                )

        # AP overshoot
        ax.plot(t[spike_idxs],
                V[spike_idxs],
                "x",
                ms=7,
                color='black',
                label='AP overshoot'
                )

        # AP widths
        ax.hlines(*width_lines,
                  color="red",
                  lw=2,
                  label='AP width'
                  )

        # AHP depths
        ax.plot(t[ahp_depth_idxs],
                V[ahp_depth_idxs],
                'o',
                ms=5,
                color='indianred',
                label='AHP depth'
                )

        # latency to first spike
        ax.hlines(self.V_rest,
                  t_stim_on,
                  t[spike_idxs[0]],
                  color='black',
                  lw=1.5,
                  ls=":"
                  )
        ax.vlines(t[spike_idxs[0]],
                  self.V_rest,
                  spike_heights[0],
                  color='black',
                  lw=1.5,
                  ls=":",
                  label="Lat. to first spike"
                  )

        # spike rate; mark spike locations
        for i in range(n_spikes):
            if i == 0:
                ax.vlines(t[spike_idxs[i]],
                          V[spike_idxs[i]],
                          48,
                          color='darkorange',
                          ls='--',
                          label='Spike rate'
                          )
            else:
                ax.vlines(t[spike_idxs[i]],
                          V[spike_idxs[i]],
                          48,
                          color='darkorange',
                          ls='--'
                          )
        # the marked ISIs are used to compute the accommodation index
        # ISI arrow legend
        ax.plot([],
                [],
                color='g',
                marker=r'$\longleftrightarrow$',
                linestyle='None',
                markersize=15,
                label='ISIs'
                )
        for i in range(n_spikes - 1):
            ax.annotate('',
                        xy=(t[spike_idxs[i]], 48),
                        xycoords='data',
                        xytext=(t[spike_idxs[i + 1]], 48),
                        textcoords='data',
                        arrowprops={'arrowstyle': '<->', 'color': 'g'}
                        )

        ax.set(xlabel='Time (ms)',
               ylabel='Membrane Potential (mV)'
               )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  labels,
                  loc='center left',
                  bbox_to_anchor=(1.04, 0.5),
                  fancybox=True,
                  borderaxespad=0.1,
                  ncol=1,
                  frameon=False
                  )


if __name__ == "__main__":
    import seaborn as sns

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

    sns.set(context="paper", style='whitegrid', rc={"axes.facecolor": "0.99"})

    class ConstantStimulus:
        def __init__(self, I_amp, t_stim_on, t_stim_off):
            self.I_amp = I_amp
            self.t_stim_on = t_stim_on
            self.t_stim_off = t_stim_off

        def __call__(self, t):
            return self.I_amp if self.t_stim_on <= t <= self.t_stim_off else 0

    T = 120     # Simulation time [ms]
    dt = 0.025  # Time step
    I_amp = 10
    t_stim_on = 20
    t_stim_off = 100

    stimulus = ConstantStimulus(I_amp, t_stim_on, t_stim_off)
    # stimulus = 10
    hh = HodgkinHuxley(stimulus, T, dt)
    V, t = hh(gbar_K=36., gbar_Na=120., noise=True, noise_seed=42)

    sps = SpikeStats(t_stim_on, t_stim_off, threshold=0)

    spike_data = sps.find_spikes(V, t)
    peaks = spike_data['spike_idxs']
    print(f'n spikes: {sps.n_spikes(V, t)}')
    print(f'peaks: {peaks}')
    print(len(peaks))

    #plt.plot(t, V)
    #plt.plot(t[peaks], V[peaks], "x")
    # plt.show()
    #t0 = 12.2
    #print(np.abs(t - t0).argmin())
    #print(t.flat[np.abs(t - t0).argmin()])

    s_stats = ["n_spikes",
               "spike_rate",
               "latency_to_first_spike",
               "average_AP_overshoot",
               "average_AHP_depth",
               "average_AP_width",
               "accommodation_index"]

    #sum_stats = hh.stats_df(t_stim_on, t_stim_off, sum_stats=s_stats)
    # print(sum_stats)

    #hh.plot_spike_statistics(t_stim_on, t_stim_off)
    hh.plot_voltage_trace(with_stim=True)
    plt.show()
