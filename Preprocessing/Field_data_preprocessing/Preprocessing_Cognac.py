import pandas as pd
import sys
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Preprocessing/ERA5_downscaling/')
from Preprocessing_functions import *
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'