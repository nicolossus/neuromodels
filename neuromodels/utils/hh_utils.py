import numpy as np


def compute_q10_correction(q10, T1, T2):
    """Compute the Q10 temperature coefficient.

    As explained in [1]_, the time course of voltage clamp recordings are
    strongly affected by temperature: the rates of activation and inactivation
    increase with increasing temperature. The :math:`Q_{10}` temperature
    coefficient, a measure of the increase in rate for a  10 :math:`^{\circ}C`
    temperature change, is a correction factor used in HH-style models to
    quantify this temperature dependence.

    In HH-style models, the adjustment due to temperature can be achieved by
    decreasing the time constants by a factor :math:`Q_{10}^{(T_2 - T_1)/10}`,
    where the temperatures :math:`T_1 < T_2`. The temperature unit must be
    either the Celsius or the Kelvin. Note that :math:`T_1` and :math:`T_2`
    must have the same unit, and do not need to be exactly 10 degrees apart.

    Parameters
    ----------
    q10 : :obj:`float`
        The :math:`Q_{10}` temperature coefficient.
    T1 : :obj:`float`
        Temperature at which the first rate is recorded.
    T2 : :obj:`float`
        Temperature at which the second rate is recorded.

    Returns
    -------
    correction : :obj:`float`
        Correction factor due to temperature.

    References
    ----------
    .. [1] D. Sterratt, B. Graham, A. Gillies, D. Willshaw,
           "Principles of Computational Modelling in Neuroscience",
           Cambridge University Press, 2011.
    """

    # that the test below allows T1 = T2 is intentional; the function should
    # accomendate for no correction, i.e. a correction factor equal to 1.
    if T1 > T2:
        msg = ("T2 must be greater than or equal to T1")
        raise ValueError(msg)
    return q10**((T2 - T1) / 10)


def vtrap(x, y):
    """Traps for zero in denominator of rate eqns.

    HH-style model rate equations often contain expressions that are
    equivalent to

    .. math::
        x / (np.exp(u) - 1),

    where :math:`u = x / y`. From Taylor series approximation, one can find that
    the above expression is approximated by

    .. math::
        y * (1 - u / 2)

    if :math:`u << 1`.

    This approximation ensures that indeterminate cases are handled properly.

    Parameters
    ----------
    x : :obj:`float` or :term:`ndarray`
        Numerator in the exponential function argument.
    y : :obj:`float` or :term:`ndarray`
        Denominator in the exponential function argument.

    Returns
    -------
    vtrap : :obj:`float` or :term:`ndarray`
        The result of the evaluation.

    References
    ----------
    Inspired by the ``vtrap`` function in the NEURON simulator;
    github.com/neuronsimulator/nrn/blob/master/src/nrnoc/hh.mod
    """

    u = x / y
    return np.where(u < 1e-6, y * (1 - u / 2), x / (np.exp(u) - 1))
