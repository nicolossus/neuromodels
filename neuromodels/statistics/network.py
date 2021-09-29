#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import elephant.statistics as es
import neo
import numpy as np
import quantities as pq
import scipy.stats

ALLOWED_STATISTICS = ["mean_firing_rate",
                      "mean_cv",
                      "fanofactor"]


'''
# assign list
l = [1, 2.0, 'have', 'a', 'geeky', 'day']

# assign string
s = 'geeky'

# check if string is present in the list
if s in l:
    print(f'{s} is present in the list')
else:
    print(f'{s} is not present in the list')
'''


class SpikeTrainStats:
    """
    The statistics calculator can be used as a callable by just providing the
    spike trains ... [REPHRASE] ... The statistics to be computed must
    then be passed to the class constructor, along with other keyword arguments
    """

    def __init__(self, stats=None, t_start=None, t_stop=None):
        # check allowed statistics
        self._stats = stats
        # error handling
        if isinstance(self._stats, (list, tuple, np.ndarray)):
            for i, stat in enumerate(self._stats):
                if not stat in ALLOWED_STATISTICS:
                    msg = (f"Unknown statistic '{stat}' provided. Refer to "
                           "documentation for a list of available statistics.")
                    raise ValueError(msg)
                self._stats[i] = "_" + stat

        self._t_start = t_start
        self._t_stop = t_stop

    def __call__(self, spiketrains):
        if self._stats is None:
            msg = ("'stats' must be provided as a list of statistic attributes"
                   " in order to make the instance callable.")
            raise ValueError(msg)
        self._spiketrains = spiketrains
        sum_stats = [getattr(self, stat) for stat in self._stats]
        return sum_stats

    @ property
    def _mean_firing_rate(self):
        return self.mean_firing_rate(self._spiketrains, self._t_start, self._t_stop)

    @ property
    def _mean_cv(self):
        return self.mean_cv(self._spiketrains)

    @ property
    def _fanofactor(self):
        return self.fanofactor(self._spiketrains)

    def _is_empty(self, spiketrains):
        """
        Returns True if an empty list is specified, or if all spike trains are empty.
        """
        # Array of spike counts (one per spike train)
        spike_counts = np.array([len(st) for st in spiketrains])

        return all(count == 0 for count in spike_counts)

    def mean_firing_rate(self, spiketrains, t_start=None, t_stop=None):
        """Compute the time and population-averaged firing rate.

        Compute the mean firing rate of input spike trains.

        The time averaged firing rate of a single spike train is calculated as
        the number of spikes in the spike train in the range [t_start, t_stop]
        divided by the time interval t_stop - t_start. 

        The mean firing rate of all the
        provided spike trains is simply calculated by averaging over all the
        recorded neurons.

        Notes
        -----
        The function uses the most common definition of a firing rate, namely the
        temporal average. Hence, the mean firing rate :math:`\bar{\nu}` of the
        spike trains is the spike count :math:`n^{\mathrm{sp}}_i` that occur over a
        time interval of length :math:`\Delta t` in spike train :math:`i`,
        :math:`i=1, ..., N_{\mathrm{spt}}`, divided by :math:`\Delta t` and
        averaged over the number of spike trains :math:`N_{\mathrm{spt}}`:

        .. math::
            \bar{\nu} = \frac{1}{N_{\mathrm{spt}}} \sum_{i=1}^{N_\mathrm{spt}} \frac{n_i^{\mathrm{sp}}}{\Delta t}

        The statistic is computed using the open-source package Elephant
        (RRID:SCR_003833); a library for the analysis of electrophysiological
        data.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.
        t_start : :obj:`float` or :obj:`pq.Quantity`, optional
            The start time to use for the time interval. If `None`, it is retrieved
            from the `t_start` attribute of :obj:`neo.SpikeTrain`. Default: `None`.
        t_stop : :obj:`float` or :obj:`pq.Quantity`, optional
            The stop time to use for the time interval. If `None`, it is retrieved
            from the `t_stop` attribute of :obj:`neo.SpikeTrain`. Default: `None`.

        Returns
        -------
        mean_firing_rate : :obj:`float`
            Firing rate averaged over input spike trains in units :math:`Hz`.
            Returns np.inf if an empty list is specified, or if all spike trains
            are empty.
        """
        firing_rates = []

        if self._is_empty(spiketrains):
            return np.inf

        for spiketrain in spiketrains:
            firing_rate = es.mean_firing_rate(spiketrain,
                                              t_start=t_start,
                                              t_stop=t_stop)
            firing_rate.units = pq.Hz
            firing_rates.append(firing_rate.magnitude)

        return np.mean(firing_rates)

    def mean_cv(self, spiketrains):
        """Compute the coefficient of variation averaged over recorded spike trains.

        The coefficient of variation (CV) is a measure of spike train variablity
        and is defined as the standard deviation of ISIs divided by their mean.
        A regularly spiking neuron would have a CV of 0, since there is no variance
        in the ISIs, whereas a Poisson process has a CV of 1.

        Notes
        -----
        CHANGE THIS

        .. math::
            \mathrm{CV} = \frac{\sqrt{\mathrm{Var}(\mathrm{ISIs})}}{\mathbb{E}[\mathrm{ISIs}]}

        The statistic is computed using the open-source package Elephant
        (RRID:SCR_003833); a library for the analysis of electrophysiological
        data.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.

        Returns
        -------
        mean_cv : :obj:`float`
            Mean coefficient of variation of spike trains. Returns np.inf if
            an empty list is specified, or if all spike trains are empty.
        """
        if self._is_empty(spiketrains):
            return np.inf

        cv = [es.cv(es.isi(spiketrain)) for spiketrain in spiketrains]

        return np.mean(cv)

    def fanofactor(self, spiketrains):
        """Compute the Fano factor of the spike counts.

        Parameters
        ----------
        spiketrains : :term:`array_like`
            Spike trains as a list of :obj:`neo.SpikeTrain` objects.

        Returns
        -------
        fanofactor: :obj:`float`
            The Fano factor of the spike counts of the input spike trains. Returns
            np.inf if an empty list is specified, or if all spike trains are empty.
        """

        if self._is_empty(spiketrains):
            return np.inf

        return es.fanofactor(spiketrains)


if __name__ == "__main__":
    import elephant
    import quantities as pq
    from elephant.spike_train_generation import (homogeneous_gamma_process,
                                                 homogeneous_poisson_process)

    spiketrain1 = homogeneous_poisson_process(rate=10 * pq.Hz,
                                              t_start=0. * pq.ms,
                                              t_stop=10000. * pq.ms)
    spiketrain2 = homogeneous_gamma_process(a=3,
                                            b=10 * pq.Hz,
                                            t_start=0. * pq.ms,
                                            t_stop=10000. * pq.ms)
    sts = SpikeTrainStats(stats=["mean_firing_rate", "mean_cv", "fanofactor"])
    spiketrains = [spiketrain1, spiketrain2]
    print(sts.mean_firing_rate(spiketrains))
    sum_stat = sts(spiketrains)
    print(sum_stat)
