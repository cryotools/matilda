import pandas as pd
import xarray as xr
from datetime import date, datetime
import os

# Setting the parameter for the MATILDA simulation
def MATILDA_parameter(df, set_up_start = None, set_up_end = None, sim_start = None, sim_end = None, freq = "D", area_cat = 0, \
                      area_glac = 0, ele_dat = 0, ele_glac = 0, ele_cat = 0, lr_temp = -0.006, lr_prec = 0, TT_snow = 0, \
                      TT_rain = 2, CFMAX_snow = 2.8, CFMAX_ice = 5.6, CFR_snow = 0.05, CFR_ice = 0.05, BETA = 1.0, \
                      CET=0.15, FC=250, K0=0.055, K1=0.055, K2=0.04, LP=0.7, MAXBAS=3.0, PERC=1.5, UZL = 120, PCORR = 1.0, \
                      SFCF = 0.7, CWH = 0.1):
    # Checking the parameters:
    if 1 > BETA or BETA > 6:
        print("WARNING: The parameter BETA exceeds boundaries.")
    if 0 > CET or CET > 0.3:
        print("WARNING: The parameter CET exceeds boundaries.")
    if 50 > FC or FC > 500:
        print("WARNING: The parameter FC exceeds boundaries.")
    if 0.01 > K0 or K0 > 0.4:
        print("WARNING: The parameter K0 exceeds boundaries.")
    if 0.01 > K1 or K1 > 0.4:
        print("WARNING: The parameter K1 exceeds boundaries.")
    if 0.001 > K2 or K2 > 0.15:
        print("WARNING: The parameter K2 exceeds boundaries.")
    if 0.3 > LP or LP > 1:
        print("WARNING: The parameter LP exceeds boundaries.")
    if 1 >= MAXBAS or MAXBAS > 7:
        print("WARNING: The parameter MAXBAS exceeds boundaries.")
        return
    if 0 > PERC or PERC > 3:
        print("WARNING: The parameter PERC exceeds boundaries.")
    if 0 > UZL or UZL > 500:
        print("WARNING: The parameter UZL exceeds boundaries.")
    if 0.5 > PCORR or PCORR > 2:
        print("WARNING: The parameter PCORR exceeds boundaries.")
    if TT_snow > TT_rain:
        print("WARNING: TT_snow is higher than TT_rain.")
    if -1.5 > TT_snow or TT_snow < 2.5:
        print("WARNING: The parameter TT_snow exceeds boundaries.")
    if -1.5 > TT_rain or TT_rain > 2.5:
        print("WARNING: The parameter TT_rain exceeds boundaries.")
    if 1 > CFMAX_ice or CFMAX_ice > 10:
        print("WARNING: The parameter CFMAX_ice exceeds boundaries.")
    if 1 > CFMAX_snow or CFMAX_snow > 10:
        print("WARNING: The parameter CFMAX_snow exceeds boundaries.")
    if 0.4 > SFCF or SFCF > 1:
        print("WARNING: The parameter SFCF exceeds boundaries.")
    if 0 > CFR_ice or CFR_ice > 0.1:
        print("WARNING: The parameter CFR_ice exceeds boundaries.")
    if 0 > CFR_snow or CFR_snow > 0.1:
        print("WARNING: The parameter CFR_snow exceeds boundaries.")
    if 0 > CWH or CWH > 0.2:
        print("WARNING: The parameter CWH exceeds boundaries.")

    if set_up_end and sim_start is not None:
        if set_up_end > sim_start:
            print("WARNING: The set up period exceeds the start of the simulation period.")
    if set_up_start is None:
        set_up_start = df["TIMESTAMP"].iloc[0]
    if set_up_end is None:
        set_up_end = pd.to_datetime(df["TIMESTAMP"].iloc[0])
        set_up_end = set_up_end + pd.DateOffset(years=1)
        set_up_end = str(set_up_end)
    if sim_start is None:
        sim_start = df["TIMESTAMP"].iloc[0]
    if sim_end is None:
        sim_end = df["TIMESTAMP"].iloc[-1]

    freq_long = ""
    if freq == "D":
        freq_long = "Daily"
    elif freq == "W":
        freq_long = "Weekly"
    elif freq == "M":
        freq_long = "Monthly"
    elif freq == "Y":
        freq_long = "Yearly"
    else:
        print("WARNING: Data frequency " + freq +" is not supported. Supported are D (daily), W (weekly), M (monthly) or Y (yearly).")

    if area_glac > area_cat:
        print("WARNING: The glacier area is bigger than the overall catchment area.")

    parameter = pd.Series({"set_up_start":set_up_start, "set_up_end":set_up_end, "sim_start":sim_start, "sim_end":sim_end, \
                           "freq":freq, "freq_long":freq_long, "area_cat":area_cat, "area_glac":area_glac, "ele_dat":ele_dat, \
                            "ele_glac":ele_glac, "ele_cat":ele_cat, "lr_temp":lr_temp, "lr_prec":lr_prec, "TT_snow":TT_snow, \
                           "TT_rain":TT_rain, "CFMAX_snow":CFMAX_snow, "CFMAX_ice":CFMAX_ice, "CFR_snow":CFR_snow, \
                           "CFR_ice":CFR_ice, "BETA":BETA,  "CET":CET, "FC":FC, "K0":K0, "K1":K1, "K2":K2, "LP":LP,  \
                           "MAXBAS":MAXBAS, "PERC":PERC, "UZL":UZL, "PCORR":PCORR, "SFCF":SFCF, "CWH":CWH})
    print("Parameter for the MATILDA simulation are set")
    return parameter

def MATILDA_preproc(df, parameter, obs=None):
    print("---")
    print("Reading in the data")
    print("Set up period between " + str(parameter.set_up_start) + " and " + str(parameter.set_up_end) + " to get appropriate initial values")
    print("Simulation period between " + str(parameter.sim_start) + " and " + str(parameter.sim_end))
    if parameter.set_up_start > parameter.sim_start:
        print("WARNING: Spin up period starts later than the simulation period")
    if isinstance(df, xr.Dataset):
        df = df.sel(time=slice(parameter.set_up_start, parameter.sim_end))
    else:
        df.set_index('TIMESTAMP', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df[parameter.set_up_start: parameter.sim_end]
    if obs is not None:
        obs.set_index('Date', inplace=True)
        obs.index = pd.to_datetime(obs.index)
        obs = obs[parameter.set_up_start: parameter.sim_end]
        obs = obs.resample("D").sum()
        # expanding the observation period to the whole one year, filling the NAs with 0
        idx_first = obs.index.year[1]
        idx_last = obs.index.year[-1]
        idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=obs.index.name)
        obs = obs.reindex(idx)
        obs = obs.fillna(0)

    if obs is not None:
        return df, obs
    if obs is None:
        return df


def MATILDA_save_output(output_MATILDA, parameter, output_path):
    output_path = output_path + parameter.sim_start[:4] + "_" + parameter.sim_end[
                                                                       :4] + "_" + datetime.now().strftime(
        "%Y-%m-%d_%H:%M:%S") + "/"
    os.mkdir(output_path)  # creating the folder to save the plots

    print("Saving the MATILDA output to disc")
    output_MATILDA[0].to_csv(output_path + "model_output_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
        output_MATILDA[0].index.values[-1])[:4] + ".csv")
    output_MATILDA[2].to_csv(output_path + "model_stats_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
        output_MATILDA[0].index.values[-1])[:4] + ".csv")
    parameter.to_csv(output_path + "model_parameter.csv")

    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[3].savefig(
            output_path + "meteorological_data_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
            dpi=output_MATILDA[3].dpi)
    else:
        output_MATILDA[3].savefig(
            output_path + "meteorological_data_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[:4] + ".png", dpi=output_MATILDA[3].dpi)
    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[4].savefig(output_path + "model_runoff_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
                                  dpi=output_MATILDA[4].dpi)
    else:
        output_MATILDA[4].savefig(
            output_path + "model_runoff_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[
                                                                                               :4] + ".png",
            dpi=output_MATILDA[4].dpi)
    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[5].savefig(output_path + "HBV_output_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
                                  dpi=output_MATILDA[5].dpi)
    else:
        output_MATILDA[5].savefig(
            output_path + "HBV_output_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[
                                                                                             :4] + ".png",
            dpi=output_MATILDA[5].dpi)
    print("---")

