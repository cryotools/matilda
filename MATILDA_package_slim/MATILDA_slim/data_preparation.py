import pandas as pd
import xarray as xr
from datetime import date


def data_preproc(df, obs, parameter):
    print("---")
    print("Reading in the data")
    print("Spin up period between " + str(parameter.cal_period_start) + " and " + str(parameter.cal_period_end))
    print("Simulation period between " + str(parameter.sim_period_start) + " and " + str(parameter.sim_period_end))
    if parameter.cal_period_start > parameter.sim_period_start:
        print("WARNING: Spin up period starts later than the simulation period")
    if isinstance(df, xr.Dataset):
        df = df.sel(time=slice(parameter.cal_period_start, parameter.sim_period_end))
    else:
        df.set_index('TIMESTAMP', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[parameter.cal_period_start: parameter.sim_period_end]
    obs.set_index('Date', inplace=True)
    obs.index = pd.to_datetime(obs.index)
    obs = obs[parameter.cal_period_start: parameter.sim_period_end]
    obs = obs.resample("D").sum()
    # expanding the observation period to the whole one year, filling the NAs with 0
    idx_first = obs.index.year[1]
    idx_last = obs.index.year[-1]
    idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=obs.index.name)
    obs = obs.reindex(idx)
    obs = obs.fillna(0)

    return df, obs
