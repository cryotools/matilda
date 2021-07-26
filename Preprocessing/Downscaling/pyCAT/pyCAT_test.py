from pycat.io import Dataset
from pycat.esd import QuantileMapping
from pycat.esd import ScaledDistributionMapping

obs = Dataset('sample-data', 'observation.nc')
mod = Dataset('sample-data', 'model*.nc')
sce = Dataset('sample-data', 'scenario*.nc')
sdm = ScaledDistributionMapping(obs, mod, sce)
sdm.correct()

obs