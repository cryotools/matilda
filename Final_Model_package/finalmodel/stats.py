import pandas as pd

# Statistical analysis of the output variables
def create_statistics(output_calibration):
    stats = output_calibration.describe()
    sum = pd.DataFrame(output_calibration.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    return stats