import neuromodels as nm
import numpy as np
import numpy.random as rng
import scipy as sp
import scipy.stats as stats

N = stats.norm(0, 1)

r = np.random.rand(0, 1)
q = rng.rand(0, 1)

I = nm.stimuli.constant()
# print(I['info'])

'''
[(10, False),
 (10.5, False),
 (lambda t: 10, False),
 (lambda t, noise=True: 10, False),
 (np.ones(5001) * 10, False),
 ({'t': 10}, True),
 (lambda t, N: 10, True),
 (np.ones(5000) * 10, True), ])
'''

stimulus = 10

T = 50
dt = 0.1
hh = nm.HodgkinHuxley()
hh.solve(stimulus, T, dt)

T = 50
dt = 0.2
print(int(T / dt) + 1)
celsius = 6.3
q10 = 3**((celsius - 6.3) / 10)
print(q10)
