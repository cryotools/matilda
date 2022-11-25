import numpy as np
import pandas as pd
import spotpy
from pathlib import Path
import os
import sys
import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'


fast_path = 'OUTPATH'
jobname = 'DB_NAME'

# Load results
os.chdir(fast_path)
results = spotpy.analyser.load_csv_results(jobname)

# Example plot to show the sensitivity index of each parameter
spotpy.analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=3, fig_name=jobname + '_plot.png')

# Example to get the sensitivity index of each parameter
SI = spotpy.analyser.get_sensitivity_of_fast(results, print_to_console=False)

parnames = spotpy.analyser.get_parameternames(results)
sens = pd.DataFrame(SI)
sens['param'] = parnames
sens.to_csv(jobname + '_sensitivity_indices.csv')
