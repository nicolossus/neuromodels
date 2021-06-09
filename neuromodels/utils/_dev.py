'''
Developed functionality that might be useful
'''

from pprint import pprint

import numpy as np
from sympy import diff, exp, limit, symbols

###############################################################################
# L'Hôpital's rule


def l_hopital(A_val, Vh_val, k_val):
    '''
    Symbolic computation of L'Hôpital's rule on rate eqns.
    '''
    Vm = symbols('Vm')   # membrane potential
    Vh = symbols('Vh')   # half-activation voltage
    A = symbols('A')     # rate constant
    k = symbols('k')     # slope of activation curve

    num = A * (Vm - Vh)
    denom = (1 - exp(-(Vm - Vh) / k))

    num = num.subs(A, A_val)
    num = num.subs(Vh, Vh_val)
    denom = denom.subs(Vh, Vh_val)
    denom = denom.subs(k, k_val)
    # expr = num / denom
    # pprint(expr)
    l_hop = limit(diff(num) / diff(denom), Vm, Vh_val)
    return l_hop


# alpha_n
A_alpha_n = 0.01
Vh_alpha_n = -55
k_alpha_n = 10

alpha_n = l_hopital(A_alpha_n, Vh_alpha_n, k_alpha_n)  # should be 0.1
print(f"{alpha_n=}")

###############################################################################
# Safe exponential function

expmax_float64 = np.log(np.finfo(np.float64).max)
expmax_clip = np.int64(expmax_float64) - np.int64(1)
expmin_clip = - expmax_clip


def safe_exp(u):
    """Safe exponential evaluation.

    Suppress exponential overflow by imposing a bounded range. This
    function may be useful if the value of :math:`\exp(x)` would overflow
    the numeric range of float64.
    """
    return np.exp(np.clip(u, expmin_clip, expmax_clip))


print('Safe exp:', safe_exp(712))
print('Standard exp:', np.exp(712))
