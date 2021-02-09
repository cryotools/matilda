import pandas as pd
import numpy as np

# Nash–Sutcliffe model efficiency coefficient
def NS(obs, model):
    nash_sut = 1 - np.sum((obs-model)**2) / (np.sum((obs-obs.mean())**2))
    if nash_sut > 1 or nash_sut < -1:
        nash_sut = "error"
    return nash_sut

# Statistical analysis of the output variables
def create_statistics(output):
    stats = output.describe()
    sum = pd.DataFrame(output.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    stats = stats.round(3)
    return stats
