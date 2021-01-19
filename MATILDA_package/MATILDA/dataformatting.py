import pandas as pd
import xarray as xr
import numpy as np
from _datetime import date

def data_preproc(df, cal_period_start, sim_period_end):
    if isinstance(df, xr.Dataset):
        df = df.sel(time=slice(cal_period_start, sim_period_end))
    elif "TIMESTAMP" in df:
        df.set_index('TIMESTAMP', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[cal_period_start: sim_period_end]
    elif 'Date' in df:
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[cal_period_start: sim_period_end]
        if "Qobs" in df.columns: # expanding the observation period to the whole one year, filling the NAs with 0
            idx_first = df.index.year[1]
            idx_last = df.index.year[-1]
            idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=df.index.name)
            df = df.reindex(idx)
            df = df.fillna(0)
    return df

def glacier_downscaling(df, height_diff, lapse_rate_temperature=0, lapse_rate_precipitation=0):
    df_DDM = df.copy()
    df_DDM["T2"] = np.where(df_DDM["T2"] <= 100, df_DDM["T2"] + 273.15, df_DDM["T2"])
    df_DDM["T2"] = df_DDM["T2"] + height_diff * float(lapse_rate_temperature)
    df_DDM["RRR"] = df_DDM["RRR"] + height_diff * float(lapse_rate_precipitation)
    return df_DDM

def output_postproc(output_hbv, output_DDM, obs):
    output = pd.concat([output_hbv, output_DDM], axis=1)
    output = pd.concat([output, obs], axis=1)
    output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
    return output

def output_cosipy(output, ds):
    output_cosipy = output[{"Qobs", "Q_Total", "DDM_smb", "DDM_total_melt"}]
    cosipy_runoff = ds.Q.mean(dim=["lat", "lon"])
    cosipy_smb = ds.surfMB.mean(dim=["lat", "lon"])
    #cosipy_smb = cosipy_smb.resample(time="D").sum(dim="time")
    cosipy_melt = ds.surfM.mean(dim=["lat", "lon"])
    #cosipy_melt = cosipy_melt.resample(time="D").sum(dim="time")
    output_cosipy["Q_COSIPY"] = cosipy_runoff.to_dataframe().Q.resample('D').sum()*1000
    output_cosipy["COSIPY_smb"] = cosipy_smb.to_dataframe().surfMB.resample('D').sum()*1000
    output_cosipy["COSIPY_melt"] = cosipy_melt.to_dataframe().surfM.resample('D').sum()*1000
    output_cosipy = output_cosipy.round(3)
    return output_cosipy

def output_parameter(parameter_HBV, parameter_DDM):
    parameter = parameter_HBV.append(parameter_DDM)
    return parameter


def plot_data(output, plot_frequency, cal_period_start, sim_period_end):
    obs = output["Qobs"].resample(plot_frequency).agg(pd.DataFrame.sum, skipna=False)
    plot_data = output.resample(plot_frequency).agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
    plot_data["Qobs"] = obs
    plot_data = plot_data[cal_period_start: sim_period_end]
    return plot_data

def plot_data_cosipy(output_cosipy, plot_frequency, cal_period_start, sim_period_end):
    plot_data_cosipy = output_cosipy.resample(plot_frequency).agg(
        {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb": "sum", "DDM_total_melt": "sum", \
         "COSIPY_smb": "sum", "COSIPY_melt": "sum"})
    plot_data_cosipy = plot_data_cosipy[cal_period_start: sim_period_end]
    return plot_data_cosipy
