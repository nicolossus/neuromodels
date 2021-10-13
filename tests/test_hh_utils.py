import os
import sys

import numpy as np  # isort:skip
import pytest  # isort:skip

test_path = os.path.dirname(os.path.abspath(__file__))  # noqa  # isort:skip
sys.path.append(test_path + '/../neuromodels')  # noqa  # isort:skip

import utils  # isort:skip


def test_compute_q10_correction():
    """Test that computation of Q10 correction factor returns expected"""
    correction = utils.compute_q10_correction(3, 3.6, 3.6)
    expected = 1.0
    assert correction == expected


def test_compute_q10_correction_raises():
    """Test that ValueError is raised if T1 > T2"""
    with pytest.raises(ValueError):
        correction = utils.compute_q10_correction(3, 6.6, 3.6)


def test_vtrap():
    """Test that vtrap handles an indeterminate case correctly"""
    V = -55
    rate = .01 * utils.vtrap(-(V + 55), 10)
    expected = 0.1
    assert rate == expected
