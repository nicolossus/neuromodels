
Welcome to NeuroModels' documentation!
=======================================

:Release: |version|
:Date: |today|

NeuroModels is a Python toolbox for simulating and analyzing computational
neuroscience models.

.. plot::
      :context: close-figs
      :format: doctest
      :include-source: False

      >>> import matplotlib.pyplot as plt
      >>> import numpy as np
      >>> x = np.linspace(0, 2*np, 100)
      >>> y = np.sin(x)
      >>> plt.plot(x, y)
      >>> plt.show()

Hodgkin-Huxley:

.. plot::
      :context: close-figs
      :format: doctest
      :include-source: False

      >>> import matplotlib.pyplot as plt
      >>> import neuromodels as nm
      >>> hh.HodgkinHuxley()
      >>> hh.solve(10, 50, 0.1)
      >>> plt.plot(hh.t, hh.V)
      >>> plt.show()

.. toctree::
   :maxdepth: 2
   :caption: User Documentation:

   installation

.. toctree::
  :maxdepth: 2
  :caption: API Reference

  neuromodels

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
