import os
import sys

import numpy as np  # isort:skip
import pytest  # isort:skip

test_path = os.path.dirname(os.path.abspath(__file__))  # noqa  # isort:skip
sys.path.append(test_path + '/../neuromodels')  # noqa  # isort:skip

import neuromodels  # isort:skip
from neuromodels.models import ODEsNotSolved  # isort:skip


def test_all_property_raises():
    """Test that a set of properties are implemented and raises.

    A large set of properties need to exist and raise a ODEsNotSolved
    error if called before solve.
    """
    props = ['t', 'V', 'n', 'm', 'h', ]

    hh = nm.HodgkinHuxley()
    raises = 0
    for prop in props:
        try:
            getattr(hh, prop)
        except ODEsNotSolved:
            raises += 1
    assert raises == len(props)


@pytest.mark.parametrize("value", ["string", [1, 2], np.array([10]), ])
def test_all_property_setter_raises(value):
    """Test that model parameters raises if wrong data type is set.

    All model parameters can be set as class attributes and raise a TypeError
    if the wrong data type is provided.
    """
    props = ['V_rest', 'Cm', 'gbar_K', 'gbar_Na',
             'gbar_L', 'E_K', 'E_Na', 'E_L', 'degC', ]
    hh = nm.HodgkinHuxley()
    raises = 0
    for prop in props:
        try:
            setattr(hh, prop, value)
        except TypeError:
            raises += 1
    assert raises == len(props)


def test_conductances_raises():
    """Test that conductances raises if set as negative number.

    Conductances must be non-negative and should raise a ValueError if
    set as negative.
    """
    props = ['gbar_K', 'gbar_Na', 'gbar_L', ]
    hh = nm.HodgkinHuxley()
    raises = 0
    for prop in props:
        try:
            setattr(hh, prop, -10)
        except ValueError:
            raises += 1
    assert raises == len(props)


@pytest.mark.parametrize("value", [0, -1])
def test_capacitance_raises(value):
    """Test that capacitance raises if not strictly positive.

    Capacitance must be strictly positive and should raise a ValueError if not.
    """
    hh = nm.HodgkinHuxley()
    with pytest.raises(ValueError):
        setattr(hh, 'Cm', value)


@pytest.mark.parametrize(('T', 'dt', 'exception'),
                         [(10, '0.1', TypeError),
                          (10, [0.1], TypeError),
                          (10, np.array([0.1]), TypeError),
                          ('10', 0.1, TypeError),
                          ([10], 0.1, TypeError),
                          (np.array([10]), 0.1, TypeError),
                          (10, -0.1, ValueError),
                          (-10, 0.1, ValueError)])
def test_solver_time_arguments(T, dt, exception):
    """Test that the solver raises if provided time arguments are wrong.

    The time arguments, T and dt, must be given as int or float larger than
    zero. The solver should raise TypeError if wrong data type and ValueError
    if the value is less than or equal to zero.
    """

    def stimulus(t):
        return 10

    hh = nm.HodgkinHuxley()
    with pytest.raises(exception):
        hh.solve(stimulus, T, dt)


@pytest.mark.parametrize(("state_param", "ic_index"),
                         [('V', 0),
                          ('n', 1),
                          ('m', 2),
                          ('h', 3), ])
def test_rest_state(state_param, ic_index):
    """Test that a resting system stays at rest.

    A system at rest (no external stimulus) should stay in the rest state.
    """

    def stimulus(t):
        return 0

    hh = nm.HodgkinHuxley()
    hh.solve(stimulus, 50, 0.01)
    observed = hh.V
    observed = getattr(hh, state_param)
    actual = np.ones(observed.shape) * hh._initial_conditions[ic_index]
    tolerance = 0.01

    assert pytest.approx(observed, tolerance) == actual


@pytest.mark.parametrize(('stimulus', 'should_raise'),
                         [(10, False),
                          (10.5, False),
                          (lambda t: 10, False),
                          (lambda t, noise=True: 10, False),
                          (np.ones(2001) * 10, False),
                          ({'t': 10}, True),
                          (lambda t, N: 10, True),
                          (np.ones(2000) * 10, True), ])
def test_stimulus(stimulus, should_raise):
    """Test that the solver raises only if stimulus is passed wrong.

    The solver should only accept stimulus as a scalar (int or float),
    a callable or numpy.ndarray. If callable, the call signature must be
    one and only one positional argument (kewword arguments are allowed).
    When passed as numpy.ndarray, stimulus must have shape (int(T/dt)+1).
    """
    T = 50
    dt = 0.025
    hh = nm.HodgkinHuxley()

    is_raised = False
    try:
        hh.solve(stimulus, T, dt)
    except Exception:
        is_raised = True

    assert should_raise == is_raised
