import neuromodels as nm
from nest import raster_plot
from neuromodels import vtrap
from neuromodels.statistics import network
#from neuromodels.statistics import fanofactor, mean_cv, mean_firing_rate
from neuromodels.utils import vtrap

bnet = nm.BrunelNetwork()
bnet.simulate()
spiketrains = bnet.spiketrains()
# print(mean_firing_rate(spiketrains))
