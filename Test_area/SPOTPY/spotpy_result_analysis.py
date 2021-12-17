import pandas as pd
from pathlib import Path
import sys
import spotpy
import numpy as np
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/SPOTPY')
##
wd = '/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/Jyrgalang_catchment/'
filename = '20211210_sceua_jyrgalang_10y_10000rep'


##
result_path = wd + filename
results = spotpy.analyser.load_csv_results(result_path)
trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]

likes = trues['like1']
maximum = np.nanmax(likes)
index = np.where(likes == maximum)

best_param = trues[index]
best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
par_names = spotpy.analyser.get_parameternames(trues)
param_zip = zip(par_names, best_param_values)
best_param = dict(param_zip)
best_param_df = pd.DataFrame(best_param, index=[0])
print(best_param_df.transpose())
print('Best objective value: ' + str(maximum))

best_param_df.transpose().to_csv(wd + 'best_param_' + filename + '_0.' + str(maximum).split('.')[1] + '.csv')




## compare:
# kysyl1 = pd.read_csv(wd + 'best_param_rope_0,7676.csv').transpose()
# kysyl2 = pd.read_csv(wd + 'best_param_sa_0,7607.csv').transpose()
# karab = pd.read_csv(wd + 'best_param_karab_sceua_0,843.csv').transpose()
#
# kysyl1.reset_index(level=0, inplace=True)
# kysyl2.reset_index(level=0, inplace=True)
# karab.reset_index(level=0, inplace=True)