#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

spiketrain_list = []
for nid in gdf_id_list:
    selected_ids = self._get_selected_ids(nid, id_column,
                                          time_column, t_start,
                                          t_stop, time_unit, data)
    times = data[selected_ids[0]:selected_ids[1], time_column]
    spiketrain_list.append(SpikeTrain(

        times, units=time_unit,
        t_start=t_start, t_stop=t_stop,
        id=nid, **args))

t_start
SpikeTrain(times, units='ms', t_start=)


def ISIs(self):
    """Interspike intervals (ISIs).

    ISI is the time between subsequent action potentials.

    Returns
    -------
    ISIs : array_like
        Interspike intervals.
    """

    ISIs = []
    for i in range(self.n_spikes - 1):
        ISI = self._time[self._spikes_ind[i + 1]] - \
            self._time[self._spikes_ind[i]]
        ISIs.append(ISI)

    return np.array(ISIs)


def isi(spiketrain, axis=-1):
    pass


'''
END
'''

cv = scipy.stats.variation


def isi(spiketrain, axis=-1):
    """
    Return an array containing the inter-spike intervals of the spike train.
    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, a `np.ndarray`, or a
    list of time spikes. If either a `neo.SpikeTrain` or `pq.Quantity` is
    provided, the return value will be `pq.Quantity`, otherwise `np.ndarray`.
    The units of `pq.Quantity` will be the same as `spiketrain`.
    Visualization of this function is covered in Viziphant:
    :func:`viziphant.statistics.plot_isi_histogram`.
    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or array-like
        The spike times.
    axis : int, optional
        The axis along which the difference is taken.
        Default: the last axis
    Returns
    -------
    intervals : np.ndarray or pq.Quantity
        The inter-spike intervals of the `spiketrain`.
    Warns
    -----
    UserWarning
        When the input array is not sorted, negative intervals are returned
        with a warning.
    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.isi([0.3, 4.5, 6.7, 9.3])
    array([4.2, 2.2, 2.6])
    """
    if isinstance(spiketrain, neo.SpikeTrain):
        intervals = np.diff(spiketrain.magnitude, axis=axis)
        # np.diff makes a copy
        intervals = pq.Quantity(intervals, units=spiketrain.units, copy=False)
    else:
        intervals = np.diff(spiketrain, axis=axis)
    if (intervals < 0).any():
        warnings.warn("ISI evaluated to negative values. "
                      "Please sort the input array.")

    return intervals


def mean_firing_rate(spiketrain, t_start=None, t_stop=None, axis=None):
    """
    Return the firing rate of the spike train.
    The firing rate is calculated as the number of spikes in the spike train
    in the range `[t_start, t_stop]` divided by the time interval
    `t_stop - t_start`. See the description below for cases when `t_start` or
    `t_stop` is None.
    Accepts a `neo.SpikeTrain`, a `pq.Quantity` array, or a plain
    `np.ndarray`. If either a `neo.SpikeTrain` or `pq.Quantity` array is
    provided, the return value will be a `pq.Quantity` array, otherwise a
    plain `np.ndarray`. The units of the `pq.Quantity` array will be the
    inverse of the `spiketrain`.
    Parameters
    ----------
    spiketrain : neo.SpikeTrain or pq.Quantity or np.ndarray
        The spike times.
    t_start : float or pq.Quantity, optional
        The start time to use for the interval.
        If None, retrieved from the `t_start` attribute of `spiketrain`. If
        that is not present, default to 0. All spiketrain's spike times below
        this value are ignored.
        Default: None
    t_stop : float or pq.Quantity, optional
        The stop time to use for the time points.
        If not specified, retrieved from the `t_stop` attribute of
        `spiketrain`. If that is not present, default to the maximum value of
        `spiketrain`. All spiketrain's spike times above this value are
        ignored.
        Default: None
    axis : int, optional
        The axis over which to do the calculation; has no effect when the
        input is a neo.SpikeTrain, because a neo.SpikeTrain is always a 1-d
        vector. If None, do the calculation over the flattened array.
        Default: None
    Returns
    -------
    float or pq.Quantity or np.ndarray
        The firing rate of the `spiketrain`
    Raises
    ------
    TypeError
        If the input spiketrain is a `np.ndarray` but `t_start` or `t_stop` is
        `pq.Quantity`.
        If the input spiketrain is a `neo.SpikeTrain` or `pq.Quantity` but
        `t_start` or `t_stop` is not `pq.Quantity`.
    ValueError
        If the input spiketrain is empty.
    Examples
    --------
    >>> from elephant import statistics
    >>> statistics.mean_firing_rate([0.3, 4.5, 6.7, 9.3])
    0.4301075268817204
    """
    if isinstance(spiketrain, neo.SpikeTrain) and t_start is None \
            and t_stop is None and axis is None:
        # a faster approach for a typical use case
        n_spikes = len(spiketrain)
        time_interval = spiketrain.t_stop - spiketrain.t_start
        time_interval = time_interval.rescale(spiketrain.units)
        rate = n_spikes / time_interval
        return rate

    if isinstance(spiketrain, pq.Quantity):
        # Quantity or neo.SpikeTrain
        if not is_time_quantity(t_start, allow_none=True):
            raise TypeError("'t_start' must be a Quantity or None")
        if not is_time_quantity(t_stop, allow_none=True):
            raise TypeError("'t_stop' must be a Quantity or None")

        units = spiketrain.units
        if t_start is None:
            t_start = getattr(spiketrain, 't_start', 0 * units)
        t_start = t_start.rescale(units).magnitude
        if t_stop is None:
            t_stop = getattr(spiketrain, 't_stop',
                             np.max(spiketrain, axis=axis))
        t_stop = t_stop.rescale(units).magnitude

        # calculate as a numpy array
        rates = mean_firing_rate(spiketrain.magnitude, t_start=t_start,
                                 t_stop=t_stop, axis=axis)

        rates = pq.Quantity(rates, units=1. / units)
    elif isinstance(spiketrain, (np.ndarray, list, tuple)):
        if isinstance(t_start, pq.Quantity) or isinstance(t_stop, pq.Quantity):
            raise TypeError("'t_start' and 't_stop' cannot be quantities if "
                            "'spiketrain' is not a Quantity.")
        spiketrain = np.asarray(spiketrain)
        if len(spiketrain) == 0:
            raise ValueError("Empty input spiketrain.")
        if t_start is None:
            t_start = 0
        if t_stop is None:
            t_stop = np.max(spiketrain, axis=axis)
        time_interval = t_stop - t_start
        if axis and isinstance(t_stop, np.ndarray):
            t_stop = np.expand_dims(t_stop, axis)
        rates = np.sum((spiketrain >= t_start) & (spiketrain <= t_stop),
                       axis=axis) / time_interval
    else:
        raise TypeError("Invalid input spiketrain type: '{}'. Allowed: "
                        "neo.SpikeTrain, Quantity, ndarray".
                        format(type(spiketrain)))
    return rates


'''
Skaar
'''


def convert_to_spiketrains(spike_arr):
    """
    Converts array of neuron IDs and spike times to list of spiketrains
    """
    id_sorted = spike_arr[spike_arr[:, 0].argsort()]
    _, first_indices = np.unique(id_sorted[:, 0], return_index=True)
    spiketrains = []
    for j, i in enumerate(first_indices):
        if i != first_indices[-1]:
            ii = first_indices[j + 1]
        else:
            ii = len(id_sorted)
        spiketrain = np.sort(id_sorted[i:ii, 1])
        spiketrains.append(spiketrain[(spiketrain > 500.0)])
    return spiketrains


spiketrains = convert_to_spiketrains(np.concatenate((ex_spikes, in_spikes)))


def calculate_cvs(spiketrain_list):
    """
    Calculates average CV from list of spiketrains
    """
    cvs = []
    for spiketrain in spiketrain_list:
        if len(spiketrain) < 3:
            continue
        isi = np.ediff1d(spiketrain)
        cvs.append(variation(isi))
    if len(cvs) == 0:
        return np.nan
    else:
        return np.mean(cvs)


'''
uncertainpy
'''


def cv(self, simulation_end, spiketrains):
    """
    Calculate the coefficient of variation for each neuron.
    Parameters
    ----------
    simulation_end : float
        The simulation end time.
    neo_spiketrains : list
        A list of Neo spiketrains.
    Returns
    -------
    time : None
    values : array
        The coefficient of variation for each spiketrain.
    """
    if len(spiketrains) == 0:
        return None, None

    cv = []
    for spiketrain in spiketrains:
        cv.append(elephant.statistics.cv(spiketrain))

    return None, np.array(cv)


def average_cv(self, simulation_end, spiketrains):
    """
    Calculate the average coefficient of variation.
    Parameters
    ----------
    simulation_end : float
        The simulation end time.
    neo_spiketrains : list
        A list of Neo spiketrains.
    Returns
    -------
    time : None
    values : float
        The average coefficient of variation of each spiketrain.
    """
    if len(spiketrains) == 0:
        return None, None

    cv = []
    for spiketrain in spiketrains:
        cv.append(elephant.statistics.cv(spiketrain))

    return None, np.mean(cv)


def average_isi(self, simulation_end, spiketrains):
    """
    Calculate the average interspike interval (isi) variation for each neuron.
    Parameters
    ----------
    simulation_end : float
        The simulation end time.
    neo_spiketrains : list
        A list of Neo spiketrains.
    Returns
    -------
    time : None
    average_isi : float
       The average interspike interval.
    """
    if len(spiketrains) == 0:
        return None, None

    isi = []
    for spiketrain in spiketrains:
        if len(spiketrain) > 1:
            isi.append(np.mean(elephant.statistics.isi(spiketrain)))

    return None, np.mean(isi)


class NetworkFeatures(GeneralNetworkFeatures):
    """
    Network features of a model result, works with all models that return
    the simulation end time, and a list of spiketrains.
    Parameters
    ----------
    new_features : {None, callable, list of callables}
        The new features to add. The feature functions have the requirements
        stated in ``reference_feature``. If None, no features are added.
        Default is None.
    features_to_run : {"all", None, str, list of feature names}, optional
        Which features to calculate uncertainties for.
        If ``"all"``, the uncertainties are calculated for all
        implemented and assigned features.
        If None, or an empty list ``[]``, no features are
        calculated.
        If str, only that feature is calculated.
        If list of feature names, all the listed features are
        calculated. Default is ``"all"``.
    interpolate : {None, "all", str, list of feature names}, optional
        Which features are irregular, meaning they have a varying number of
        time points between evaluations. An interpolation is performed on
        each irregular feature to create regular results.
        If ``"all"``, all features are interpolated.
        If None, or an empty list, no features are interpolated.
        If str, only that feature is interpolated.
        If list of feature names, all listed features are interpolated.
        Default is None.
    labels : dictionary, optional
        A dictionary with key as the feature name and the value as a list of
        labels for each axis. The number of elements in the list corresponds
        to the dimension of the feature. Example:
        .. code-block:: Python
            new_labels = {"0d_feature": ["x-axis"],
                          "1d_feature": ["x-axis", "y-axis"],
                          "2d_feature": ["x-axis", "y-axis", "z-axis"]
                         }
    units : {None, Quantities unit}, optional
        The Quantities unit of the time in the model. If None, ms is used.
        The default is None.
    instantaneous_rate_nr_samples : int
        The number of samples used to calculate the instantaneous rate.
        Default is 50.
    isi_bin_size : int
        The size of each bin in the ``binned_isi`` method.
        Default is 1.
    corrcoef_bin_size : int
        The size of each bin in the ``corrcoef`` method.
        Default is 1.
    covariance_bin_size : int
        The size of each bin in the ``covariance`` method.
        Default is 1.
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging is performed.
        Default logger level is "info".
    Attributes
    ----------
    features_to_run : list
        Which features to calculate uncertainties for.
    interpolate : list
        A list of irregular features to be interpolated.
    utility_methods : list
        A list of all utility methods implemented. All methods in this class
        that is not in the list of utility methods is considered to be a feature.
    labels : dictionary
        Labels for the axes of each feature, used when plotting.
    logger : logging.Logger
        Logger object responsible for logging to screen or file.
    instantaneous_rate_nr_samples : int
        The number of samples used to calculate the instantaneous rate.
        Default is 50.
    isi_bin_size : int
        The size of each bin in the ``binned_isi`` method.
        Default is 1.
    corrcoef_bin_size : int
        The size of each bin in the ``corrcoef`` method.
        Default is 1.
    covariance_bin_size : int
        The size of each bin in the ``covariance`` method.
        Default is 1.
    Notes
    -----
    Implemented features are:
    ======================= ======================= =======================
    cv                      average_cv              average_isi,
    local_variation mean    local_variation         average_firing_rate
    instantaneous_rate      fanofactor              van_rossum_dist
    victor_purpura_dist     binned_isi              corrcoef
    covariance
    ======================= ======================= =======================
    All features in this set of features take the following input arguments:
    simulation_end : float
        The simulation end time
    neo_spiketrains : list
        A list of Neo spiketrains.
    The model must return:
    simulation_end : float
        The simulation end time
    spiketrains : list
        A list of spiketrains, each spiketrain is a list of the times when
        a given neuron spikes.
    Raises
    ------
    ImportError
        If elephant or quantities is not installed.
    See also
    --------
    uncertainpy.features.Features.reference_feature : reference_feature showing the requirements of a feature function.
    """

    def __init__(self,
                 new_features=None,
                 features_to_run="all",
                 interpolate=None,
                 labels={},
                 units=None,
                 instantaneous_rate_nr_samples=50,
                 isi_bin_size=1,
                 corrcoef_bin_size=1,
                 covariance_bin_size=1,
                 logger_level="info"):

        if not prerequisites:
            raise ImportError(
                "Network features require: elephant and quantities")

        if units is None:
            units = pq.ms

        unit_string = str(units).split()[1]

        implemented_labels = {"cv": ["Neuron nr", "Coefficient of variation"],
                              "average_cv": ["Average coefficient of variation"],
                              "average_isi": ["Average interspike interval ({})".format(unit_string)],
                              "local_variation": ["Neuron nr", "Local variation"],
                              "average_local_variation": ["Mean local variation"],
                              "average_firing_rate": ["Neuron nr", "Rate (Hz)"],
                              "instantaneous_rate": ["Time (ms)", "Neuron nr", "Rate (Hz)"],
                              "fanofactor": ["Fanofactor"],
                              "van_rossum_dist": ["Neuron nr", "Neuron nr", ""],
                              "victor_purpura_dist": ["Neuron nr", "Neuron nr", ""],
                              "binned_isi": ["Interspike interval ({})".format(unit_string),
                                             "Neuron nr", "Count"],
                              "corrcoef": ["Neuron nr", "Neuron nr", "Correlation coefficient"],
                              "covariance": ["Neuron nr", "Neuron nr", "Covariance"]
                              }

        implemented_labels.update(labels)

        super(NetworkFeatures, self).__init__(new_features=new_features,
                                              features_to_run=features_to_run,
                                              interpolate=interpolate,
                                              labels=implemented_labels,
                                              units=units)

        self.instantaneous_rate_nr_samples = instantaneous_rate_nr_samples
        self.isi_bin_size = isi_bin_size
        self.corrcoef_bin_size = corrcoef_bin_size
        self.covariance_bin_size = covariance_bin_size

    def cv(self, simulation_end, spiketrains):
        """
        Calculate the coefficient of variation for each neuron.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        values : array
            The coefficient of variation for each spiketrain.
        """
        if len(spiketrains) == 0:
            return None, None

        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.array(cv)

    def average_cv(self, simulation_end, spiketrains):
        """
        Calculate the average coefficient of variation.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        values : float
            The average coefficient of variation of each spiketrain.
        """
        if len(spiketrains) == 0:
            return None, None

        cv = []
        for spiketrain in spiketrains:
            cv.append(elephant.statistics.cv(spiketrain))

        return None, np.mean(cv)

    def binned_isi(self, simulation_end, spiketrains):
        """
        Calculate a histogram of the interspike interval.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : array
            The center of each bin.
        binned_isi : array
            The binned interspike intervals.
        """
        if len(spiketrains) == 0:
            return None, None

        binned_isi = []
        bins = np.arange(
            0, spiketrains[0].t_stop.magnitude + self.isi_bin_size, self.isi_bin_size)

        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi = elephant.statistics.isi(spiketrain)
                binned_isi.append(np.histogram(isi, bins=bins)[0])

            else:
                binned_isi.append(np.zeros(len(bins) - 1))

        centers = bins[1:] - 0.5
        return centers, binned_isi

    def average_isi(self, simulation_end, spiketrains):
        """
        Calculate the average interspike interval (isi) variation for each neuron.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        average_isi : float
           The average interspike interval.
        """
        if len(spiketrains) == 0:
            return None, None

        isi = []
        for spiketrain in spiketrains:
            if len(spiketrain) > 1:
                isi.append(np.mean(elephant.statistics.isi(spiketrain)))

        return None, np.mean(isi)

    def local_variation(self, simulation_end, spiketrains):
        """
        Calculate the measure of local variation.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        local_variation : list
            The local variation for each spiketrain.
        """
        if len(spiketrains) == 0:
            return None, None

        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))
            else:
                local_variation.append(None)

        return None, local_variation

    def average_local_variation(self, simulation_end, spiketrains):
        """
        Calculate the average of the local variation.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        average_local_variation : float
            The average of the local variation for each spiketrain.
        """
        if len(spiketrains) == 0:
            return None, None

        local_variation = []
        for spiketrain in spiketrains:
            isi = elephant.statistics.isi(spiketrain)
            if len(isi) > 1:
                local_variation.append(elephant.statistics.lv(isi))

        return None, np.mean(local_variation)

    def average_firing_rate(self, simulation_end, spiketrains):
        """
        Calculate the mean firing rate.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        average_firing_rate : float
            The mean firing rate of all neurons.
        """
        average_firing_rates = []

        if len(spiketrains) == 0:
            return None, None

        for spiketrain in spiketrains:
            average_firing_rate = elephant.statistics.mean_firing_rate(
                spiketrain)
            average_firing_rate.units = pq.Hz
            average_firing_rates.append(average_firing_rate.magnitude)

        return None, average_firing_rates

    def instantaneous_rate(self, simulation_end, spiketrains):
        """
        Calculate the mean instantaneous firing rate.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : array
            Time of the instantaneous firing rate.
        instantaneous_rate : float
            The instantaneous firing rate.
        """
        if len(spiketrains) == 0:
            return None, None

        instantaneous_rates = []
        t = None
        for spiketrain in spiketrains:
            if len(spiketrain) > 2:
                sampling_period = spiketrain.t_stop / self.instantaneous_rate_nr_samples
                # try/except to solve problem with elephant
                try:
                    instantaneous_rate = elephant.statistics.instantaneous_rate(
                        spiketrain, sampling_period)
                    instantaneous_rates.append(
                        np.array(instantaneous_rate).flatten())

                    if t is None:
                        t = instantaneous_rate.times.copy()
                        t.units = self.units
                except TypeError:
                    instantaneous_rates.append(None)

            else:
                instantaneous_rates.append(None)

        if t is None:
            return None, instantaneous_rates
        else:
            return t.magnitude, instantaneous_rates

    def fanofactor(self, simulation_end, spiketrains):
        """
        Calculate the fanofactor.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        fanofactor : float
            The fanofactor.
        """
        if len(spiketrains) == 0:
            return None, None

        return None, elephant.statistics.fanofactor(spiketrains)

    def van_rossum_dist(self, simulation_end, spiketrains):
        """
        Calculate van Rossum distance.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        van_rossum_dist : 2D array
            The van Rossum distance.
        """
        if len(spiketrains) == 0:
            return None, None

        van_rossum_dist = elephant.spike_train_dissimilarity.van_rossum_dist(
            spiketrains)

        # van_rossum_dist returns 0.j imaginary parts in some cases
        van_rossum_dist = np.real_if_close(van_rossum_dist)
        if np.any(np.iscomplex(van_rossum_dist)):
            return None, None

        return None, van_rossum_dist

    def victor_purpura_dist(self, simulation_end, spiketrains):
        """
        Calculate the Victor-Purpura's distance.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        values : 2D array
            The Victor-Purpura's distance.
        """
        if len(spiketrains) == 0:
            return None, None

        victor_purpura_dist = elephant.spike_train_dissimilarity.victor_purpura_dist(
            spiketrains)

        return None, victor_purpura_dist

    def corrcoef(self, simulation_end, spiketrains):
        """
        Calculate the pairwise Pearson's correlation coefficients.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        values : 2D array
            The pairwise Pearson's correlation coefficients.
        """
        if len(spiketrains) == 0:
            return None, None

        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.corrcoef_bin_size * self.units)
        corrcoef = elephant.spike_train_correlation.corrcoef(binned_sts)

        return None, corrcoef

    def covariance(self, simulation_end, spiketrains):
        """
        Calculate the pairwise covariances.
        Parameters
        ----------
        simulation_end : float
            The simulation end time.
        neo_spiketrains : list
            A list of Neo spiketrains.
        Returns
        -------
        time : None
        values : 2D array
            The pairwise covariances.
        """
        if len(spiketrains) == 0:
            return None, None

        binned_sts = elephant.conversion.BinnedSpikeTrain(spiketrains,
                                                          binsize=self.covariance_bin_size * self.units)
        covariance = elephant.spike_train_correlation.covariance(binned_sts)

        return None, covariance
