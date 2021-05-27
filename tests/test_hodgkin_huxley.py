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
