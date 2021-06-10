#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect

import numpy as np
from neuromodels.utils import compute_q10_correction, vtrap
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class ODEsNotSolved(Exception):
    """Failed attempt at accessing solutions.

    A call to the ODE systems solve method must be
    carried out before the solution properties
    can be used.
    """
    pass


class HodgkinHuxley:
    r"""Class for representing the original Hodgkin-Huxley model.

    The Hodgkin–Huxley model describes how action potentials in neurons are
    initiated and propagated. From a biophysical point of view, action
    potentials are the result of currents that pass through ion channels in
    the cell membrane. In an extensive series of experiments on the giant axon
    of the squid, Hodgkin and Huxley succeeded to measure these currents and
    to describe their dynamics in terms of differential equations.

    This class implements the original Hodgkin-Huxley model for the
    sodium, potassium and leakage channels found in the squid giant axon
    membrane. Membrane voltage is in absolute mV and has been reversed in
    polarity from the original HH convention and shifted to reflect a resting
    potential of -65 mV.

    All model parameters can be accessed (get or set) as class attributes.
    Solutions are available as class attributes after calling the class method
    :meth:`solve`.

    Parameters
    ----------
    V_rest : :obj:`float`
        Resting potential of neuron in units :math:`mV`, default=-65.0.
    Cm : :obj:`float`
        Membrane capacitance in units :math:`\mu F/cm^2`, default=1.0.
    gbar_K : :obj:`float`
        Potassium conductance in units :math:`mS/cm^2`, default=36.0.
    gbar_Na : :obj:`float`
        Sodium conductance in units :math:`mS/cm^2`, default=120.0.
    gbar_L : :obj:`float`
        Leak conductance in units :math:`mS/cm^2`, default=0.3.
    E_K : :obj:`float`
        Potassium reversal potential in units :math:`mV`, default=-77.0.
    E_Na : :obj:`float`
        Sodium reversal potential in units :math:`mV`, default=50.0.
    E_L : :obj:`float`
        Leak reversal potential in units :math:`mV`, default=-54.4.
    degC : :obj:`float`
        Temperature when recording (should be to set a squid-appropriate
        temperature) in units :math:`^{\circ}C`, default=6.3.

    Attributes
    ----------
    V_rest : :obj:`float`
        **Model parameter:** Resting potential.
    Cm : :obj:`float`
        **Model parameter:** Membrane capacitance.
    gbar_K : :obj:`float`
        **Model parameter:** Potassium conductance.
    gbar_Na : :obj:`float`
        **Model parameter:** Sodium conductance.
    gbar_L : :obj:`float`
        **Model parameter:** Leak conductance.
    E_K : :obj:`float`
        **Model parameter:** Potassium reversal potential.
    E_Na : :obj:`float`
        **Model parameter:** Sodium reversal potential.
    E_L : :obj:`float`
        **Model parameter:** Leak reversal potential.
    degC : :obj:`float`
        **Model parameter:** Temperature when recording in degrees Celsius.
    t : :term:`ndarray`
        **Solution:** Array of time points ``t``.
    V : :term:`ndarray`
        **Solution:** Array of voltage values ``V`` at ``t``.
    n : :term:`ndarray`
        **Solution:** Array of state variable values ``n`` at ``t``.
    m : :term:`ndarray`
        **Solution:** Array of state variable values ``m`` at ``t``.
    h : :term:`ndarray`
        **Solution:** Array of state variable values ``h`` at ``t``.

    Notes
    -----
    Default parameter values as given by Hodgkin and Huxley [1]_.

    References
    ----------
    .. [1] A. L. Hodgkin, A. F. Huxley, "A quantitative description of membrane
           current and its application to conduction and excitation in nerve",
           J. Physiol. 117, pp. 500-544, 1952.

    Examples
    --------

    .. plot::

        import matplotlib.pyplot as plt
        import neuromodels as nm

        # Initialize the Hodgkin-Huxley system; model parameters can either
        # be set in the constructor or accessed as class attributes:
        hh = nm.HodgkinHuxley(V_rest=-70)
        hh.gbar_K = 36

        # The simulation parameters needed are the simulation time T, the time
        # step dt, and the input stimulus, the latter either as a constant,
        # callable with call signature `(t)` or ndarray with `shape=(int(T/dt)+1,)`:
        T = 50.
        dt = 0.025

        def stimulus(t):
            return 10 if 10 <= t <= 40 else 0

        # The system is solved by calling the class method `solve` and the
        # solutions can be accessed as class attributes:
        hh.solve(stimulus, T, dt)
        t = hh.t
        V = hh.V

        plt.plot(t, V)
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.show()
    """

    def __init__(self, V_rest=-65., Cm=1., gbar_K=36., gbar_Na=120.,
                 gbar_L=0.3, E_K=-77., E_Na=50., E_L=-54.4, degC=6.3):

        # Hodgkin-Huxley model parameters
        self._V_rest = V_rest      # resting potential [mV]
        self._Cm = Cm              # membrane capacitance [μF/cm**2]
        self._gbar_K = gbar_K      # potassium conductance [mS/cm**2]
        self._gbar_Na = gbar_Na    # sodium conductance [mS/cm**2]
        self._gbar_L = gbar_L      # leak coductance [mS/cm**2]
        self._E_K = E_K            # potassium reversal potential [mV]
        self._E_Na = E_Na          # sodium reversal potential [mV]
        self._E_L = E_L            # leak reversal potential [mV]
        self._degC = degC          # temperature [degrees Celsius]

        # Temperature coefficient (correction factor)
        self._q10 = compute_q10_correction(q10=3, T1=6.3, T2=self._degC)

    def __call__(self, t, y):
        r"""RHS of the Hodgkin-Huxley ODEs.

        Parameters
        ----------
        t : float
            The time point.
        y : tuple of floats
            A tuple of the state variables, ``y = (V, n, m, h)``.
        """

        V, n, m, h = y

        dVdt = (self.I_inj(t) - self.I_K(V, n) -
                self.I_Na(V, m, h) - self.I_L(V)) / self._Cm
        dndt = (self.n_inf(V) - n) / self.tau_n(V)
        dmdt = (self.m_inf(V) - m) / self.tau_m(V)
        dhdt = (self.h_inf(V) - h) / self.tau_h(V)

        return [dVdt, dndt, dmdt, dhdt]

    # Membrane currents
    def I_K(self, V, n):
        """Potassium current.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.
        n : :obj:`float` or :term:`ndarray`
            Potassium channel state variable.

        Returns
        -------
        I_K : :obj:`float` or :term:`ndarray`
            Potassium current.
        """

        return self._gbar_K * (n**4) * (V - self._E_K)

    def I_Na(self, V, m, h):
        """Sodium current.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.
        m : :obj:`float` or :term:`ndarray`
            Sodium channel activation state variable.
        h : :obj:`float` or :term:`ndarray`
            Sodium channel inactivation state variable.

        Returns
        -------
        I_Na : :obj:`float` or :term:`ndarray`
            Sodium current.
        """

        return self._gbar_Na * (m**3) * h * (V - self._E_Na)

    def I_L(self, V):
        """Leak current.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        I_L : :obj:`float` or :term:`ndarray`
            Leak current.
        """
        return self._gbar_L * (V - self._E_L)

    # K channel kinetics
    def alpha_n(self, V):
        """Potassium channel activation forward reaction rate.

        Uses the :func:`~neuromodels.utils.vtrap` function to trap for zero in
        denominator of rate equation.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return .01 * vtrap(-(V + 55), 10)

    def beta_n(self, V):
        """Potassium channel activation backward reaction rate.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return .125 * np.exp(-(V + 65) / 80)

    # Na channel kinetics (activating)
    def alpha_m(self, V):
        """Sodium channel activation forward reaction rate.

        Uses the :func:`~neuromodels.utils.vtrap` function to trap for zero in
        denominator of rate equation.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return .1 * vtrap(-(V + 40), 10)

    def beta_m(self, V):
        """Sodium channel activation backward reaction rate.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return 4 * np.exp(-(V + 65) / 18.)

    # Na channel kinetics (inactivating)
    def alpha_h(self, V):
        """Sodium channel inactivation forward reaction rate.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return 0.07 * np.exp(-(V + 65) / 20.)

    def beta_h(self, V):
        """Sodium channel inactivation backward reaction rate.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        rate : :obj:`float` or :term:`ndarray`
            Reaction rate.
        """
        return 1. / (1 + np.exp(-(V + 35) / 10.))

    # steady-states and time constants
    def n_inf(self, V):
        """Potassium channel activation steady state.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        steady_state : :obj:`float` or :term:`ndarray`
            Steady state.
        """
        return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    def tau_n(self, V):
        """Potassium channel activation time constant.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        time_constant : :obj:`float` or :term:`ndarray`
            Time constant.
        """
        return 1. / (self._q10 * (self.alpha_n(V) + self.beta_n(V)))

    def m_inf(self, V):
        """Sodium channel activation steady state.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        steady_state : :obj:`float` or :term:`ndarray`
            Steady state.
        """
        return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    def tau_m(self, V):
        """Sodium channel activation time constant.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        time_constant : :obj:`float` or :term:`ndarray`
            Time constant.
        """
        return 1. / (self._q10 * (self.alpha_m(V) + self.beta_m(V)))

    def h_inf(self, V):
        """Sodium channel inactivation steady state.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        steady_state : :obj:`float` or :term:`ndarray`
            Steady state.
        """
        return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

    def tau_h(self, V):
        """Sodium channel inactivation time constant.

        Parameters
        ----------
        V : :obj:`float` or :term:`ndarray`
            Membrane potential.

        Returns
        -------
        time_constant : :obj:`float` or :term:`ndarray`
            Time constant.
        """
        return 1. / (self._q10 * (self.alpha_h(V) + self.beta_h(V)))

    @property
    def _initial_conditions(self):
        """Default Hodgkin-Huxley model initial conditions."""
        n0 = self.n_inf(self.V_rest)
        m0 = self.m_inf(self.V_rest)
        h0 = self.h_inf(self.V_rest)
        return (self.V_rest, n0, m0, h0)

    def solve(self, stimulus, T, dt, y0=None, method='RK45', **kwargs):
        r"""Solve the Hodgkin-Huxley equations.

        The equations are solved on the interval ``(0, T]`` and the solutions
        evaluted at a given interval. The solutions are not returned, but
        stored as class attributes.

        If multiple calls to solve are made, they are treated independently,
        with the newest one overwriting any old solution data.

        The solver only accepts ``stimulus`` as either a scalar
        (:obj:`int` or :obj:`float`), :obj:`callable` or :term:`ndarray`. If
        :obj:`callable`, the call signature must be one and only one positional
        argument, e.g. ``(t)``. Kewword arguments are allowed in passed
        callables. When passed as :term:`ndarray`, stimulus must have shape
        ``(int(T/dt)+1).``

        Parameters
        ----------
        stimulus : {:obj:`int`, :obj:`float`}, :obj:`callable` or :term:`ndarray`, shape=(int(T/dt)+1,)
            Input stimulus in units :math:`\mu A/cm^2`. If callable, the call
            signature must be ``(t)``.
        T : :obj:`float`
            End time in milliseconds (:math:`ms`).
        dt : :obj:`float`
            Time step where solutions are evaluated.
        y0 : :term:`array_like`, shape=(4,)
            Initial state of state variables ``V``, ``n``, ``m``, ``h``. If None,
            the default Hodgkin-Huxley model's initial conditions will be used;
            :math:`y_0 = (V_0, n_0, m_0, h_0) = (V_{rest}, n_\infty(V_0), m_\infty(V_0), h_\infty(V_0))`.
        method : :obj:`str`
            Integration method to use. Description from the
            :obj:`scipy.integrate.solve_ivp` documentation:
                * 'RK45' (default): Explicit Runge-Kutta method of order 5(4) [1]_.
                  The error is controlled assuming accuracy of the fourth-order
                  method, but steps are taken using the fifth-order accurate
                  formula (local extrapolation is done). A quartic interpolation
                  polynomial is used for the dense output [2]_. Can be applied in
                  the complex domain.
                * 'RK23': Explicit Runge-Kutta method of order 3(2) [3]_. The error
                  is controlled assuming accuracy of the second-order method, but
                  steps are taken using the third-order accurate formula (local
                  extrapolation is done). A cubic Hermite polynomial is used for the
                  dense output. Can be applied in the complex domain.
                * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
                  Python implementation of the "DOP853" algorithm originally
                  written in Fortran [14]_. A 7-th order interpolation polynomial
                  accurate to 7-th order is used for the dense output.
                  Can be applied in the complex domain.
                * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of
                  order 5 [4]_. The error is controlled with a third-order accurate
                  embedded formula. A cubic polynomial which satisfies the
                  collocation conditions is used for the dense output.
                * 'BDF': Implicit multi-step variable-order (1 to 5) method based
                  on a backward differentiation formula for the derivative
                  approximation [5]_. The implementation follows the one described
                  in [6]_. A quasi-constant step scheme is used and accuracy is
                  enhanced using the NDF modification. Can be applied in the
                  complex domain.
                * 'LSODA': Adams/BDF method with automatic stiffness detection and
                  switching [7]_, [8]_. This is a wrapper of the Fortran solver
                  from ODEPACK.
        **kwargs
            Arbitrary keyword arguments are passed along to
            :obj:`scipy.integrate.solve_ivp`.

        Notes
        -----
        The ODEs are solved numerically using the function :obj:`scipy.integrate.solve_ivp`.

        If ``stimulus`` is passed as an array, it and the time array, defined by
        ``T`` and ``dt``, will be used to create an interpolation function via
        :obj:`scipy.interpolate.interp1d`.

        ``solve_ivp`` is an ODE solver with adaptive step size. If the keyword
        argument ``first_step`` is not specified, the solver will empirically
        select an initial step size with the function ``select_initial_step``
        (found here https://github.com/scipy/scipy/blob/master/scipy/integrate/_ivp/common.py#L64).

        This function calculates two proposals and returns the smallest. It first
        calculates an intermediate proposal, ``h0``, that is based on the initial
        condition (``y0``) and the ODE's RHS evaluated for the initial condition
        (``f0``). For the standard Hodgkin-Huxley model, however, this estimated
        step size will be very large due to unfortunate circumstances (because ``norm(y0) > 0``
        while ``norm(f0) ~= 0``). Since ``h0`` only is an intermediate calculation,
        it is not used or returned by the solver. However, it is used to calculate
        the next proposal, ``h1``, by calling the RHS. Normally, this procedure
        poses no problem, but can fail if an object with a limited interval is
        present in the RHS, such as an ``interp1d`` object.

        In the case of the standard Hodgkin-Huxley model, one might be tempted
        to pass the stimulus as an array to the solver. In order for ``solve_ivp``
        to be able to evaluate the stimulus, it must be passed as a callable or
        constant. Thus, if an array is passed to the solver, an interpolation
        function must be created, in this implementation done with ``interp1d``,
        for ``solve_ivp`` to be able to evaluate it. For the reasons explained
        above, the program will hence terminate unless the ``first_step`` keyword
        is specified and is set to a sufficiently small value. In this
        implementation, ``first_step=dt`` is already set in ``solve_ivp``.

        The ``solve_ivp`` keyword ``max_step`` should be considered to be specified
        for stimuli over short time spans, in order to ensure that the solver
        does not step over them.

        Note that ``first_step`` still needs to specified even if ``max_step`` is.
        ``select_initial_step`` will be called regardless if ``first_step`` is not
        specified, and the calls for calculating h1 will be done before checking
        whether ``h0`` is larger than than the max allowed step size or not. Thus
        will only specifying ``max_step`` still result in program termination if
        ``stimulus`` is passed as an array. (Will not be a problem in this
        implementation since ``first_step`` is already specified.)

        References
        ----------
        .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
               formulae", Journal of Computational and Applied Mathematics, Vol. 6,
               No. 1, pp. 19-26, 1980.
        .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
               of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
        .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
               Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
        .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
               Stiff and Differential-Algebraic Problems", Sec. IV.8.
        .. [5] `Backward Differentiation Formula
                <https://en.wikipedia.org/wiki/Backward_differentiation_formula>`_
                on Wikipedia.
        .. [6] L. F. Shampine, M. W. Reichelt, "THE MATLAB ODE SUITE", SIAM J. SCI.
               COMPUTE., Vol. 18, No. 1, pp. 1-22, January 1997.
        .. [7] A. C. Hindmarsh, "ODEPACK, A Systematized Collection of ODE
               Solvers," IMACS Transactions on Scientific Computation, Vol 1.,
               pp. 55-64, 1983.
        .. [8] L. Petzold, "Automatic selection of methods for solving stiff and
               nonstiff systems of ordinary differential equations", SIAM Journal
               on Scientific and Statistical Computing, Vol. 4, No. 1, pp. 136-148,
               1983.
        """

        # error-handling
        self._check_solver_input(dt, 'dt')
        self._check_solver_input(T, 'T')

        # times at which to store the computed solutions
        t_eval = np.arange(0, T + dt, dt)

        if y0 is None:
            # use default HH initial conditions
            y0 = self._initial_conditions

        # handle the passed stimulus
        if isinstance(stimulus, (int, float)):
            self.I_inj = lambda t: stimulus

        elif callable(stimulus):
            sig = inspect.signature(stimulus)
            free_params_in_signature = 0
            for param in sig.parameters.values():
                if (param.kind == param.POSITIONAL_OR_KEYWORD and
                        param.default is param.empty):
                    free_params_in_signature += 1
            if free_params_in_signature > 1:
                msg = ("Callable can only take one positional argument.")
                raise TypeError(msg)

            self.I_inj = stimulus

        elif isinstance(stimulus, np.ndarray):
            if not stimulus.shape == t_eval.shape:
                msg = ("stimulus numpy.ndarray must have shape (int(T/dt)+1)")
                raise ValueError(msg)
            # Interpolate stimulus
            self.I_inj = interp1d(x=t_eval, y=stimulus)  # linear spline

        else:
            msg = ("'stimulus' must be either a scalar (int or float), "
                   "callable function of t or a numpy.ndarray of shape "
                   "(int(T/dt)+1)")
            raise TypeError(msg)

        # solve HH ODEs
        solution = solve_ivp(self,
                             t_span=(0, T),
                             y0=y0,
                             t_eval=t_eval,
                             first_step=dt,
                             method=method,
                             **kwargs)

        # store solutions
        self._time = solution.t
        self._V = solution.y[0]
        self._n = solution.y[1]
        self._m = solution.y[2]
        self._h = solution.y[3]

    # Check user input
    def _check_type_int_float(self, parameter, name):
        if not isinstance(parameter, (int, float)):
            msg = (f"{name} must be set as an int or float.")
            raise TypeError(msg)

    def _check_conductances(self, parameter, name):
        self._check_type_int_float(parameter, name)
        if parameter < 0:
            msg = ("Conductances must be non-negative.")

    def _check_solver_input(self, parameter, name):
        if not isinstance(parameter, (int, float)):
            msg = (f"{name} must be set as an int or float.")
            raise TypeError(msg)

        if parameter <= 0:
            msg = (f"{name} > 0 is required")
            raise ValueError(msg)

    # Get and set model parameters
    @property
    def V_rest(self):
        return self._V_rest

    @V_rest.setter
    def V_rest(self, V_rest):
        self._check_type_int_float(V_rest, 'V_rest')
        self._V_rest = V_rest

    @property
    def Cm(self):
        return self._Cm

    @Cm.setter
    def Cm(self, Cm):
        self._check_type_int_float(Cm, 'Cm')
        if Cm <= 0:
            msg = ("Capacitance must be strictly positive.")
        self._Cm = Cm

    @property
    def gbar_K(self):
        return self._gbar_K

    @gbar_K.setter
    def gbar_K(self, gbar_K):
        self._check_conductances(gbar_K, 'gbar_K')
        self._gbar_K = gbar_K

    @property
    def gbar_Na(self):
        return self._gbar_Na

    @gbar_Na.setter
    def gbar_Na(self, gbar_Na):
        self._check_conductances(gbar_Na, 'gbar_Na')
        self._gbar_Na = gbar_Na

    @property
    def gbar_L(self):
        return self._gbar_L

    @gbar_L.setter
    def gbar_L(self, gbar_L):
        self._check_conductances(gbar_L, 'gbar_L')
        self._gbar_L = gbar_L

    @property
    def E_K(self):
        return self._E_K

    @E_K.setter
    def E_K(self, E_K):
        self._check_type_int_float(E_K, 'E_K')
        self._E_K = E_K

    @property
    def E_Na(self):
        return self._E_Na

    @E_Na.setter
    def E_Na(self, E_Na):
        self._check_type_int_float(E_Na, 'E_Na')
        self._E_Na = E_Na

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, E_L):
        self._check_type_int_float(E_L, 'E_L')
        self._E_L = E_L

    @property
    def degC(self):
        return self._degC

    @degC.setter
    def degC(self, degC):
        self._check_type_int_float(degC, 'degC')
        self._degC = degC
        # Recompute correction factor
        self._q10 = compute_q10_correction(q10=3, T1=6.3, T2=self._degC)

    # Solutions
    @property
    def t(self):
        try:
            return self._time
        except AttributeError as e:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def V(self):
        try:
            return self._V
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def n(self):
        try:
            return self._n
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def m(self):
        try:
            return self._m
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @property
    def h(self):
        try:
            return self._h
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")
