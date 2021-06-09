import neuromodels as nm
import numpy as np
import pytest


def test_compute_q10_correction():
    """Test that computation of Q10 correction factor returns expected"""
    correction = nm.utils.compute_q10_correction(3, 3.6, 3.6)
    expected = 1.0
    assert correction == expected


def test_compute_q10_correction_raises():
    """Test that ValueError is raised if T1 > T2"""
    with pytest.raises(ValueError):
        correction = nm.utils.compute_q10_correction(3, 6.6, 3.6)


def test_vtrap():
    """Test that vtrap handles an indeterminate case correctly"""
    V = -55
    rate = .01 * nm.utils.vtrap(-(V + 55), 10)
    expected = 0.1
    assert rate == expected
