import pandas as pd
import numpy as np

# Nash–Sutcliffe model efficiency coefficient
def NS(obs, model):
    return 1 - np.sum((obs-model)**2) / (np.sum((obs-obs.mean())**2))

# Statistical analysis of the output variables
def create_statistics(output_calibration):
    stats = output_calibration.describe()
    sum = pd.DataFrame(output_calibration.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    stats = stats.round(3)
    return stats
