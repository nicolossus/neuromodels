import neuromodels as nm
import numpy as np
import pytest
from neuromodels.hodgkin_huxley import ODEsNotSolved


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
             'gbar_L', 'E_K', 'E_Na', 'E_L']
    hh = nm.HodgkinHuxley()
    raises = 0
    for prop in props:
        try:
            setattr(hh, prop, value)
        except TypeError:
            raises += 1
    assert raises == len(props)


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
