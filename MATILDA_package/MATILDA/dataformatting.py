import pandas as pd

def data_preproc(df, obs, cal_period_start, sim_period_end):
    df.set_index('TIMESTAMP', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[cal_period_start: sim_period_end]

    obs.set_index('Date', inplace=True)
    obs.index = pd.to_datetime(obs.index)
    obs = obs[cal_period_start: sim_period_end]
    return df, obs

def glacier_downscaling(df, height_diff, lapse_rate_temperature=0, lapse_rate_precipitation=0):
    df_DDM = df.copy()
    df_DDM["T2"] = df_DDM["T2"] + height_diff * float(lapse_rate_temperature)
    df_DDM["RRR"] = df_DDM["RRR"] + height_diff * float(lapse_rate_precipitation)
    return df_DDM

def output_postproc(output_hbv, output_DDM, obs):
    output = pd.concat([output_hbv, output_DDM], axis=1)
    output = pd.concat([output, obs], axis=1)
    output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
    return output

def plot_data(output, plot_frequency, cal_period_start, sim_period_end):
    plot_data = output.resample(plot_frequency).agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
    plot_data = plot_data[cal_period_start: sim_period_end]
    return plot_data