#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

ALLOWED_STATISTICS = ["n_spikes",
                      "spike_rate",
                      "latency_to_first_spike",
                      "average_AP_overshoot",
                      "average_AHP_depth",
                      "average_AP_width",
                      "accommodation_index"]


class SpikeStats:
    """

        threshold : :obj:`int` or :obj:`float`, optional
            Required height of spikes. If `None`, the value is retrived from
            the threshold attribute from the constructor (which defaults to `0`).
            Default: `None`.
    """

    def __init__(
        self,
        t_stim_on,
        t_stim_off,
        threshold=0,
        stats=None,
    ):
        # check ALLOWED_STATISTIC
        self._stats_provided = False

        # error handling
        if isinstance(stats, (list, tuple, np.ndarray)):
            self._stats_provided = True
            self._stats = stats[:]
            for i, stat in enumerate(self._stats):
                if not stat in ALLOWED_STATISTICS:
                    msg = (f"Unknown statistic '{stat}' provided. Refer to "
                           "documentation for a list of available statistics.")
                    raise ValueError(msg)
                self._stats[i] = "_" + stat

        self._t_stim_on = t_stim_on
        self._t_stim_off = t_stim_off
        self._duration = self._t_stim_off - self._t_stim_on
        self._threshold = threshold

    def __call__(self, V, t):
        if not self._stats_provided:
            msg = ("keyword argument 'stats' must be provided as a list of "
                   "statistic attributes in order to make the instance "
                   "callable.")
            raise ValueError(msg)

        self._V = V
        self._t = t
        self._spikes = self.find_spikes(self._V, self._t)
        #self._n_spikes = self._spikes["n_spikes"]

        sum_stats = [getattr(self, stat) for stat in self._stats]
        return sum_stats

    def find_spikes(self, V, t):
        """Find spikes in voltage trace.

        Find spikes in voltage trace that are above a specified threshold
        height.

        Parameters
        ----------
        V : :term:`array_like`
            The voltage array of the voltage trace.
        t : :term:`array_like`
            The time array of the voltage trace.

        Returns
        -------
        spikes : :obj:`dict`
            Dictionary containing the:
            * spike heights, key: `'spike_heights'`
            * spike times, key: `'spike_times'`
            * number of spikes, key: `'n_spikes'`
            * spike widths, key: `'spike_widths'`
            * spike positions in terms of indicies in voltage array, key: `'spike_idxs'`
            * spike widths data in terms of indices in voltage array, key: `'spike_widths_data'`
        """

        # find spikes in voltage trace
        spike_idxs, properties = find_peaks(V, height=self._threshold)

        # obtain data about width of spikes; note that returned widths will be
        # in terms of index positions in voltage array
        spike_widths_data = peak_widths(V, spike_idxs, rel_height=0.5)

        # retrieve left and right interpolated positions specifying horizontal
        # start and end positions of the found spike widths
        left_ips, right_ips = spike_widths_data[2:]

        # membrane potential as interpolation function
        V_interpolate = interp1d(x=np.linspace(0, len(V), len(V)),
                                 y=V)

        # voltage spike width in terms of physical units (instead of position
        # in array)
        spike_widths = V_interpolate(right_ips) - V_interpolate(left_ips)

        # gather results in a dictionary
        spikes = {"spike_heights": properties["peak_heights"],
                  "spike_times": t[spike_idxs],
                  "n_spikes": len(spike_idxs),
                  "spike_widths": spike_widths,
                  "spike_idxs": spike_idxs,
                  "spike_widths_data": spike_widths_data
                  }

        return spikes

    def width_lines(self, V, t):
        """Data for contour lines at which the widths where calculated.

        The purpose of this function is to prepare spike width data for
        plotting in terms of physical units.

        Can be used for plotting the located spike widths via e.g.:
            >>> features = SpikingFeatures(V, t, stim_duration, t_stim_on)
            >>> plt.hlines(*features.width_lines)

        Returns
        -------
        width_lines : 3-tuple of ndarrays
            Contour lines.
        """

        spikes = self.find_spikes(V, t)
        spike_widths_data = spikes["spike_widths_data"]

        # retrieve left and right interpolated positions specifying horizontal
        # start and end positions of the found spike widths
        left_ips, right_ips = spike_widths_data[2:]

        # time as interpolation function
        time_interpolate = interp1d(x=np.linspace(0, len(t), len(t)),
                                    y=t)

        # interpolated width positions in terms of physical units
        left_ips_physical = time_interpolate(left_ips)
        right_ips_physical = time_interpolate(right_ips)

        # the height of the contour lines at which the widths where evaluated
        width_heights = spike_widths_data[1]

        # assemble data of contour lines at which the widths where calculated
        spike_widths_data_physical = (width_heights,
                                      left_ips_physical,
                                      right_ips_physical)

        return spike_widths_data_physical

    def AHP_depth_positions(self, V, t):
        """Array of positions of after hyperpolarization depths in terms of
        indices.

        Returns
        -------
        AHP_depth_position : array_like
            The positions of the minimum voltage values between consecutive
            action potentials.
        """

        spikes = self.find_spikes(V, t)
        spike_idxs = spikes["spike_idxs"]

        ahp_depth_positions = []
        for i in range(spikes["n_spikes"] - 1):
            # search for minimum value between two spikes
            spike1_idx = spike_idxs[i]
            spike2_idx = spike_idxs[i + 1]
            # slice out voltage trace between the spikes
            V_slice = V[spike1_idx:spike2_idx]
            # find index of min. value relative to first spike
            min_rel_pos = np.argmin(V_slice)
            # the position in total voltage trace
            min_pos = spike1_idx + min_rel_pos
            ahp_depth_positions.append(min_pos)

        return ahp_depth_positions

    def isi(self, spike_times):
        """Interspike intervals (ISIs).

        ISI is the time between subsequent action potentials.

        Returns
        -------
        ISIs : array_like
            Interspike intervals.
        """

        isi = [spike_times[i + 1] - spike_times[i]
               for i in range(len(spike_times) - 1)]

        return np.array(isi)

    def n_spikes(self, V, t):
        """The number of spikes in voltage trace.

        Parameters
        ----------
        V : :term:`array_like`
            The voltage array of the voltage trace.

        Returns
        -------
        n_spikes : :obj:`int`
            Number of spikes.
        """

        spikes = self.find_spikes(V, t)
        return spikes["n_spikes"]

    def spike_rate(self, V, t):
        """Compute the spike rate.

        The spike rate, or the action potential firing rate, is the number of
        action potentials (spikes) divided by stimulus duration.

        Parameters
        ----------
        V : :term:`array_like`
            The voltage array of the voltage trace.

        Returns
        -------
        spike_rate : :obj:`float`
            Spike rate.
        """

        spikes = self.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 1:
            return np.inf

        return n_spikes / self._duration

    def latency_to_first_spike(self, V, t):
        """
        """
        spikes = self.find_spikes(V, t)

        if spikes["n_spikes"] < 1:
            return np.inf

        return spikes["spike_times"][0] - self._t_stim_on

    def average_AP_overshoot(self, V, t):
        """
        """
        spikes = self.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 1:
            return np.inf

        return np.sum(spikes["spike_heights"]) / n_spikes

    def average_AHP_depth(self, V, t):
        """
        """
        spikes = self.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]
        spike_idxs = spikes["spike_idxs"]

        if n_spikes < 3:
            return np.inf

        sum_ahp_depth = sum([np.min(V[spike_idxs[i]:spike_idxs[i + 1]])
                             for i in range(n_spikes - 1)])

        avg_ahp_depth = sum_ahp_depth / n_spikes

        return avg_ahp_depth

    def average_AP_width(self, V, t):
        """
        """
        spikes = self.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 1:
            return np.inf

        return np.sum(spikes["spike_widths"]) / n_spikes

    def accommodation_index(self, V, t):
        """

        Excerpt from Druckmann et al. (2007):
        k determines the number of ISIs that will be disregarded in order not
        to take into account possible transient behavior as observed in
        Markram et al. (2004). A reasonable value for k is either four ISIs or
        one-fifth of the total number of ISIs, whichever is the smaller of the
        two."
        """
        spikes = self.find_spikes(V, t)
        n_spikes = spikes["n_spikes"]

        if n_spikes < 2:
            return np.inf

        isi = self.isi(spikes["spike_times"])
        k = min(4, int(len(isi) / 5))

        A = 0
        for i in range(k + 1, n_spikes - 1):
            A += (isi[i] - isi[i - 1]) / (isi[i] + isi[i - 1])

        return A / (n_spikes - k - 1)

    @property
    def _n_spikes(self):
        return self._spikes["n_spikes"]

    @property
    def _spike_rate(self):
        if self._n_spikes < 1:
            return np.inf

        return self._n_spikes / self._duration

    @property
    def _latency_to_first_spike(self):
        """
        """

        if self._n_spikes < 1:
            return np.inf

        return self._spikes["spike_times"][0] - self._t_stim_on

    @property
    def _average_AP_overshoot(self):
        """
        """
        if self._n_spikes < 1:
            return np.inf

        return np.sum(self._spikes["spike_heights"]) / self._n_spikes

    @property
    def _average_AHP_depth(self):
        """
        """

        spike_idxs = self._spikes["spike_idxs"]

        if self._n_spikes < 3:
            return np.inf

        sum_ahp_depth = sum([np.min(self._V[spike_idxs[i]:spike_idxs[i + 1]])
                             for i in range(self._n_spikes - 1)])

        avg_ahp_depth = sum_ahp_depth / self._n_spikes

        return avg_ahp_depth

    @property
    def _average_AP_width(self):
        """
        """

        if self._n_spikes < 1:
            return np.inf

        return np.sum(self._spikes["spike_widths"]) / self._n_spikes

    @property
    def _accommodation_index(self):
        """

        """

        if self._n_spikes < 2:
            return np.inf

        isi = self.isi(self._spikes["spike_times"])
        k = min(4, int(len(isi) / 5))

        A = 0
        for i in range(k + 1, self._n_spikes - 1):
            A += (isi[i] - isi[i - 1]) / (isi[i] + isi[i - 1])

        return A / (self._n_spikes - k - 1)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
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

    # remove top and right axis from plots
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # HH simulation
    hh = nm.solvers.HodgkinHuxleySolver()

    # simulation parameters

    # simulation parameters

    T = 120.
    dt = 0.01
    I_amp = 0.35  # 0.1 #0.31
    t_stim_on = 10
    t_stim_off = 110
    r_soma = 40  # 15

    # input stimulus
    stimulus = nm.stimulus.constant(I_amp=I_amp,
                                    T=T,
                                    dt=dt,
                                    t_stim_on=t_stim_on,
                                    t_stim_off=t_stim_off,
                                    r_soma=r_soma)
    # print(stimulus["info"])
    I = stimulus["I"]
    I_stim = stimulus["I_stim"]

    hh.solve(I, T, dt)
    V = hh.V
    t = hh.t

    # Spike statistics, callable instance
    stats = ["spike_rate",
             "latency_to_first_spike",
             "average_AP_overshoot",
             "average_AHP_depth",
             "average_AP_width",
             "accommodation_index"]

    sps = SpikeStats(t_stim_on=t_stim_on, t_stim_off=t_stim_off, stats=stats)
    sum_stats = sps(V, t)
    print(sum_stats)

    # Spike statistics, class methods
    print("SPIKE STATS")
    sps = SpikeStats(t_stim_on=t_stim_on, t_stim_off=t_stim_off)

    # number of spikes
    n_spikes = sps.n_spikes(V, t)
    print(f"{n_spikes=}")

    # spike rate
    spike_rate = sps.spike_rate(V, t)
    print(f"{spike_rate=:.4f} mHz")

    # latency to first spike
    latency_to_first_spike = sps.latency_to_first_spike(V, t)
    print(f"{latency_to_first_spike=:.4f} ms")

    # average AP overshoot
    average_AP_overshoot = sps.average_AP_overshoot(V, t)
    print(f"{average_AP_overshoot=:.4f} mV")

    # average AHP depth
    average_AHP_depth = sps.average_AHP_depth(V, t)
    print(f"{average_AHP_depth=:.4f} mV")

    # average AP width
    average_AP_width = sps.average_AP_width(V, t)
    print(f"{average_AP_width=:.4f} mV")

    # accommodation index
    accommodation_index = sps.accommodation_index(V, t)
    print(f"{accommodation_index=:.4f}")

    # plot voltage trace with features
    spikes = sps.find_spikes(V, t)
    spike_idxs = spikes["spike_idxs"]
    spike_heights = spikes["spike_heights"]
    width_lines = sps.width_lines(V, t)
    ahp_depth_idxs = sps.AHP_depth_positions(V, t)

    fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=140)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0])

    # voltage trace
    plt.plot(t, V, lw=1.5, label='Voltage trace')

    # AP overshoot
    plt.plot(t[spike_idxs], V[spike_idxs], "x",
             ms=7, color='black', label='AP overshoot')

    # AP widths
    plt.hlines(*width_lines, color="red", lw=2, label='AP width')

    # AHP depths
    plt.plot(t[ahp_depth_idxs], V[ahp_depth_idxs], 'o',
             ms=5, color='indianred', label='AHP depth')

    # latency to first spike
    plt.hlines(hh.V_rest, t_stim_on,
               t[spike_idxs[0]], color='black', lw=1.5, ls=":")
    plt.vlines(t[spike_idxs[0]], hh.V_rest, spike_heights[0],
               color='black', lw=1.5, ls=":", label="Latency to first spike")

    # the marked ISIs are used to compute the accommodation index
    # ISI arrow legend
    plt.plot([], [], color='g', marker=r'$\longleftrightarrow$',
             linestyle='None', markersize=15, label='ISIs')

    # ISI spike 1 -> 2
    plt.vlines(t[spike_idxs[0]], V[spike_idxs[0]],
               48, color='darkorange', ls='--', label='Spike rate')
    plt.annotate('', xy=(t[spike_idxs[0]], 48), xycoords='data',
                 xytext=(t[spike_idxs[1]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})

    # ISI spike 2 -> 3
    plt.vlines(t[spike_idxs[1]], V[spike_idxs[1]],
               48, color='darkorange', ls='--')
    plt.annotate('', xy=(t[spike_idxs[1]], 48), xycoords='data',
                 xytext=(t[spike_idxs[2]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})

    # ISI spike 3 -> 4
    plt.vlines(t[spike_idxs[2]], V[spike_idxs[2]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_idxs[2]], 48), xycoords='data',
                 xytext=(t[spike_idxs[3]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 4 -> 5
    plt.vlines(t[spike_idxs[3]], V[spike_idxs[3]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_idxs[3]], 48), xycoords='data',
                 xytext=(t[spike_idxs[4]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 5 -> 6
    plt.vlines(t[spike_idxs[4]], V[spike_idxs[4]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.vlines(t[spike_idxs[5]], V[spike_idxs[5]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_idxs[4]], 48), xycoords='data',
                 xytext=(t[spike_idxs[5]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})

    plt.ylabel('Voltage (mV)')
    plt.ylim(-90, 60)
    ax.set_xticks([])
    ax.set_yticks([-80, -20, 40])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles,
               labels,
               loc='center left',
               bbox_to_anchor=(1.04, 0.5),
               fancybox=True,
               borderaxespad=0.1,
               ncol=1
               )

    ax = plt.subplot(gs[1])
    plt.plot(t, I_stim, 'k', lw=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Stimulus (nA)')
    ax.set_xticks([0, np.max(t) / 2, np.max(t)])
    ax.set_yticks([0, np.max(I_stim)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    fig.suptitle("Spike Statistics")
    plt.show()
