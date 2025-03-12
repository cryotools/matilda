# -*- coding: UTF-8 -*-
"""
MATILDA Calibration Module "mspot"
--------------------------
This script provides tools for calibrating the MATILDA glacio-hydrological model using statistical parameter optimization
techniques. It makes extensive use of the SPOTPY package for sampling, optimization, and parameter uncertainty analysis. The module
is tailored for glacierized catchments and includes support for both catchment-wide and glacier-only calibration.

Key Features:
-------------
- Parameter Sampling: Flexible setup for sampling hydrological model parameters using various algorithms from SPOTPY.
- Objective Functions: Includes multiple objective functions such as Kling-Gupta Efficiency (KGE), Mean Absolute Error
  (MAE), and custom user-defined functions.
- Calibration Targets: Supports calibration against observed discharge, glacier mass balance, and snow water equivalent.
- Statistical Analysis: Provides tools to analyze parameter posterior distributions and derive parameter bounds.
- Seasonal and Annual Analysis: Includes functions for evaluating seasonal (winter/summer) and annual metrics.
- Parallel Computing: Supports parallel sampling for computational efficiency.

Functions:
----------
1. `yesno`: Interactive function to confirm actions.
2. `dict2bounds`: Converts a dictionary of parameter values into bounds.
3. `loglike_kge`: Calculates a pseudo-log-likelihood using the Kling-Gupta Efficiency (KGE).
4. `winter`, `summer`, `annual`: Extract seasonal or annual data slices for evaluation.
5. `scaled_pdd`, `scaled_ndd`: Apply temperature lapse rates to calculate positive or negative degree days.
6. `spot_setup` and `spot_setup_glacier`: Set up SPOTPY sampling for catchment-wide and glacier-specific calibration.
7. `analyze_results`: Analyze SPOTPY results to identify the best-performing parameter set and generate diagnostic plots.
8. `psample`: Automates parameter sampling and optimization with various SPOTPY algorithms.
9. `load_parameters`: Loads the best-performing parameter set from SPOTPY results.
10. `get_par_bounds`: Derives parameter bounds from SPOTPY sampling results.

Reference:
-----------
SPOTPY Library:
   - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
     Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
   - GitHub repository: https://github.com/thouska/spotpy

Dependencies
------------
- Python 3.7 or higher
- spotpy
- pandas
- numpy
- matplotlib
- hydroeval
- scipy

Usage:
------
This script is designed to be part of the MATILDA framework as calibration tool for the matilda.core model.

License:
--------
This software is released under the MIT License. See LICENSE file for details.

Contact
-------
For questions or contributions, please contact:
- Developer: Phillip Schuster
- Email: phillip.schuster@geo.hu-berlin.de
- Institution: Humboldt-Universität zu Berlin
"""

import sys
import os
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spotpy
import HydroErr as he
from scipy.stats import gamma
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import mae

from matilda.core import (
    matilda_simulation,
    matilda_parameter,
    matilda_preproc,
    create_lookup_table,
    updated_glacier_melt,
)


class HiddenPrints:
    """
    Suppress prints during multiple iterations or noisy function calls.

    This utility class redirects standard output to `/dev/null`, effectively silencing
    any print statements during its usage. It can be used as a context manager to suppress
    output temporarily.

    Example:
        with HiddenPrints():
            # Any print statements here will be suppressed
            noisy_function()

    Methods
    -------
    __enter__()
        Redirects standard output to `/dev/null`.
    __exit__(exc_type, exc_val, exc_tb)
        Restores the original standard output.
    """

    def __init__(self):
        # Initialize the attribute in the constructor
        self._original_stdout = None

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout:
            sys.stdout.close()
        sys.stdout = self._original_stdout


def yesno(question):
    """
    Prompt the user with a yes/no question and return the response.

    This function repeatedly prompts the user with the given question until a valid
    response ('y' or 'n') is entered.

    Parameters
    ----------
    question : str
        The yes/no question to present to the user.

    Returns
    -------
    bool
        Returns True if the user responds with 'y', and False if the user responds with 'n'.
    """

    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def dict2bounds(p_dict, drop=None):
    """
    Convert a parameter dictionary to a bounds dictionary to be passed to spot_setup().

    This function takes a dictionary of parameters and converts it into a dictionary
    containing both lower (`_lo`) and upper (`_up`) bounds for each parameter. Optionally,
    specific parameters can be excluded from the resulting dictionary.

    Parameters
    ----------
    p_dict : dict
        A dictionary containing parameter names as keys and their initial values as values.
    drop : list, optional
        A list of parameter names to exclude from the resulting bounds dictionary.

    Returns
    -------
    dict
        A dictionary containing both lower (`_lo`) and upper (`_up`) bounds for the remaining
        parameters in `p_dict`.
    """
    if drop is None:
        drop = []

    for i in drop:
        p_dict.pop(i, None)  # Use pop to avoid KeyError if key doesn't exist

    p = {
        **dict(zip([i + "_lo" for i in p_dict.keys()], p_dict.values())),
        **dict(zip([i + "_up" for i in p_dict.keys()], p_dict.values())),
    }
    return p


def loglike_kge(qsim, qobs):
    """
    Calculate a pseudo log-likelihood function using the Kling-Gupta Efficiency (KGE) and a gamma distribution.

    This function computes a log-likelihood value based on the KGE between simulated and observed flow values,
    and models the exceedance deviation (ED) using a gamma distribution.

    Parameters
    ----------
    qsim : array_like
        Array of simulated flow values.
    qobs : array_like
        Array of observed flow values.

    Returns
    -------
    float
        The calculated value of the log-likelihood function.

    Notes
    -----
    The log-likelihood function is based on the Kling-Gupta Efficiency (KGE) following Liu et al. (2022).
    The exceedance deviation (ED) is derived from the KGE as `ED = 1 - KGE`. The gamma probability density
    function (PDF) is then used to model the likelihood, with shape parameter `a=0.5` and scale parameter `scale=1`.

    References
    ----------
    - Liu, Y., Fernández-Ortega, J., Mudarra, M., and Hartmann, A. (2022). "Pitfalls and a feasible solution for
      using KGE as an informal likelihood function in MCMC methods: DREAM(ZS) as an example." HESS, 26(20).
    - Kling, H., Gupta, H., & Yilmaz, K. K. (2012). "Model selection criteria for rainfall-runoff
      models: Representation of variability in Bayesian Model Averaging." Water Resources Research, 48(6), W06306.
    - `hydroeval`: https://github.com/ThibHlln/hydroeval
    - `scipy.stats.gamma`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """

    # Calculate KGE
    n = len(qobs)
    KGE = he.kge_2012(qsim, qobs, remove_zero=True)
    ED = 1 - KGE

    # Calculate log-likelihood function
    gammapdf = gamma.pdf(ED, a=0.5, scale=1)
    log_L = 0.5 * n * np.log(gammapdf)

    return log_L


def winter(year, data):
    """
    Extract the winter period for a given year from a record of the World Glacier Monitoring service (WGMS).

    This function filters the input dataset for the specified year and returns a slice object representing the winter
    period based on the `BEGIN_PERIOD` and `END_WINTER` columns.

    Parameters
    ----------
    year : int
        The year for which the winter period is to be extracted.
    data : pandas.DataFrame
        The input dataset containing at least the following columns:
        - `BEGIN_PERIOD`: The start of the winter period.
        - `END_WINTER`: The end of the winter period.
        - Index representing years.

    Returns
    -------
    slice
        A slice object representing the winter period for the specified year.

    Notes
    -----
    The function assumes that the dataset is indexed by years and contains the `BEGIN_PERIOD` and `END_WINTER` columns
    with appropriate datetime or timestamp values.

    References
    ----------
    - WGMS (2024): Fluctuations of Glaciers Database. World Glacier Monitoring Service, Zurich, Switzerland.
    https://doi.org/10.5904/wgms-fog-2024-01
    """

    data = data[data.index == year]
    winter_slice = slice(data.BEGIN_PERIOD.squeeze(), data.END_WINTER.squeeze())
    return winter_slice


def summer(year, data):
    """
    Extract the summer period for a given year from a record of the World Glacier Monitoring service (WGMS).

    This function filters the input dataset for the specified year and returns a slice object representing the summer
    period based on the `END_WINTER` and `END_PERIOD` columns.

    Parameters
    ----------
    year : int
        The year for which the summer period is to be extracted.
    data : pandas.DataFrame
        The input dataset containing at least the following columns:
        - `END_WINTER`: The end of the winter period.
        - `END_PERIOD`: The end of the summer period.
        - Index representing years.

    Returns
    -------
    slice
        A slice object representing the summer period for the specified year.

    Notes
    -----
    The function assumes that the dataset is indexed by years and contains the `END_WINTER` and `END_PERIOD` columns
    with appropriate datetime or timestamp values.

    References
    ----------
    - WGMS (2024): Fluctuations of Glaciers Database. World Glacier Monitoring Service, Zurich, Switzerland.
    https://doi.org/10.5904/wgms-fog-2024-01
    """

    data = data[data.index == year]
    summer_slice = slice(data.END_WINTER.squeeze(), data.END_PERIOD.squeeze())
    return summer_slice


def annual(year, data):
    """
    Extract the annual period for a given year from a record of the World Glacier Monitoring service (WGMS).

    This function filters the input dataset for the specified year and returns a slice object representing the
    annual period based on the `BEGIN_PERIOD` and `END_PERIOD` columns.

    Parameters
    ----------
    year : int
        The year for which the annual period is to be extracted.
    data : pandas.DataFrame
        The input dataset containing at least the following columns:
        - `BEGIN_PERIOD`: The start of the annual period.
        - `END_PERIOD`: The end of the annual period.
        - Index representing years.

    Returns
    -------
    slice
        A slice object representing the annual period for the specified year.

    Notes
    -----
    The function assumes that the dataset is indexed by years and contains the `BEGIN_PERIOD` and `END_PERIOD` columns
    with appropriate datetime or timestamp values.

    References
    ----------
    - WGMS (2024): Fluctuations of Glaciers Database. World Glacier Monitoring Service, Zurich, Switzerland.
    https://doi.org/10.5904/wgms-fog-2024-01
    """

    data = data[data.index == year]
    annual_slice = slice(data.BEGIN_PERIOD.squeeze(), data.END_PERIOD.squeeze())
    return annual_slice


def scaled_pdd(data, elev, lr):
    """
    Calculate scaled positive degree days (PDD) based on temperature lapse rate and elevation adjustments.

    This function adjusts the input temperature data for elevation differences using a given lapse rate, then calculates
    the positive degree days by setting all negative or zero temperatures to zero.

    Parameters
    ----------
    data : array_like
        Array or series of temperature values (in °C).
    elev : float
        Elevation difference (in meters) used for scaling the temperature values.
    lr : float
        Temperature lapse rate (in °C/m) used for scaling.

    Returns
    -------
    pdd : array_like
        Scaled positive degree days (PDD), where negative values are replaced with zero.

    Notes
    -----
    Positive degree days are commonly used in snow and ice melt models to estimate melt based on temperature inputs.
    This function adjusts the temperature for elevation differences before computing PDD.

    Examples
    --------
    >>> data = np.array([2, -1, 3, -2, 5])
    >>> elev = 500  # meters
    >>> lr = -0.006  # °C/m
    >>> scaled_pdd(data, elev, lr)
    array([5, 0, 6, 0, 8])
    """

    s = data + elev * lr
    pdd = np.where(s > 0, s, 0)
    return pdd


def scaled_ndd(data, elev, lr):
    """
    Calculate scaled negative degree days (NDD) based on temperature lapse rate and elevation adjustments.

    This function adjusts the input temperature data for elevation differences using a given lapse rate, then calculates
    the negative degree days by setting all positive or zero temperatures to zero and marking negative temperatures as 1.

    Parameters
    ----------
    data : array_like
        Array or series of temperature values (in °C).
    elev : float
        Elevation difference (in meters) used for scaling the temperature values.
    lr : float
        Temperature lapse rate (in °C/m) used for scaling.

    Returns
    -------
    ndd : array_like
        Scaled negative degree days (NDD), where negative temperatures are marked as 1 and others as 0.

    Notes
    -----
    Negative degree days are used in some hydrological and glaciological models to track periods with freezing conditions.
    This function adjusts the temperature for elevation differences before computing NDD.

    Examples
    --------
    >>> data = np.array([2, -1, 3, -2, 5])
    >>> elev = 500  # meters
    >>> lr = -0.006  # °C/m
    >>> scaled_ndd(data, elev, lr)
    array([0, 1, 0, 1, 0])
    """

    s = data + elev * lr
    ndd = np.where(s < 0, 1, 0)
    return ndd


def spot_setup(
    set_up_start=None,
    set_up_end=None,
    sim_start=None,
    sim_end=None,
    freq="D",
    lat=None,
    area_cat=None,
    area_glac=None,
    ele_dat=None,
    ele_glac=None,
    ele_cat=None,
    glacier_profile=None,
    elev_rescaling=True,
    target_mb=None,
    target_swe=None,
    swe_scaling=None,
    fix_param=None,
    fix_val=None,
    obj_func=None,
    lr_temp_lo=-0.0065,
    lr_temp_up=-0.0055,
    lr_prec_lo=0,
    lr_prec_up=0.002,
    BETA_lo=1,
    BETA_up=6,
    CET_lo=0,
    CET_up=0.3,
    FC_lo=50,
    FC_up=500,
    K0_lo=0.01,
    K0_up=0.4,
    K1_lo=0.01,
    K1_up=0.4,
    K2_lo=0.001,
    K2_up=0.15,
    LP_lo=0.3,
    LP_up=1,
    MAXBAS_lo=2,
    MAXBAS_up=7,
    PERC_lo=0,
    PERC_up=3,
    UZL_lo=0,
    UZL_up=500,
    PCORR_lo=0.5,
    PCORR_up=2,
    TT_snow_lo=-1.5,
    TT_snow_up=1.5,
    TT_diff_lo=0.5,
    TT_diff_up=2.5,
    CFMAX_snow_lo=0.5,
    CFMAX_snow_up=10,
    CFMAX_rel_lo=1.2,
    CFMAX_rel_up=2,
    SFCF_lo=0.4,
    SFCF_up=1,
    CWH_lo=0,
    CWH_up=0.2,
    AG_lo=0,
    AG_up=1,
    CFR_lo=0.05,
    CFR_up=0.25,
    interf=4,
    freqst=2,
):
    """
    Spotpy-based setup for parameter optimization and calibration of the MATILDA model.

    This class provides the framework to optimize the parameters of the MATILDA model using SPOTPY, a statistical parameter
    optimization tool. It enables flexible parameter setup, handles fixed parameters, and evaluates simulation results
    against observed data (e.g., streamflow, snow water equivalent, surface mass balance).

    Parameters
    ----------
    General inputs:
    - set_up_start, set_up_end : str
        Start and end dates of the setup period.
    - sim_start, sim_end : str
        Start and end dates of the simulation period.
    - freq : str, optional
        Frequency of the data, default is daily ("D").
    - lat : float, optional
        Latitude of the catchment for extraterrestrial radiation calculation.
    - area_cat, area_glac : float
        Catchment and glacierized areas in km².
    - ele_dat, ele_glac, ele_cat : float
        Mean elevations (m a.s.l.) of the data, glacier, and catchment.
    - glacier_profile : DataFrame, optional
        Glacier profile for scaling and elevation rescaling.
    - elev_rescaling : bool, optional
        Flag for enabling elevation rescaling, default is True.
    - target_mb, target_swe : float, optional
        Target surface mass balance (MB) or snow water equivalent (SWE) for calibration.
    - swe_scaling : float, optional
        Scaling factor for snow water equivalent. Used to account for the mismatch in resolution of the SWE calibration
        data and the glacier mask used for MATILDA.

    Parameter bounds for calibration:
    - Fix parameter bounds (e.g., `lr_temp_lo`, `lr_temp_up` for lapse rates).
    - `fix_param` : list, optional
        List of parameters to fix during calibration.
    - `fix_val` : dict, optional
        Fixed parameter values (if applicable).

    SPOTPY settings:
    - `interf` : int, optional
        Inference factor for parameter iterations, default is 4.
    - `freqst` : int, optional
        Frequency step for sensitivity analysis, default is 2.

    Methods
    -------
    simulation(x, param_names, fix_param, fix_val, swe_scaling)
        Executes the MATILDA simulation with sampled parameters and fixed parameters (if specified).
    evaluation()
        Prepares the observed data (e.g., streamflow, SWE) for comparison with simulation results.
    objectivefunction(simulation, evaluation, params=None)
        Defines the objective function for model evaluation using the Kling-Gupta Efficiency (KGE) or user-defined metrics.

    Returns
    -------
    spot_setup_class : class
        Configured SPOTPY setup class for MATILDA parameter optimization and calibration.

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy

    - Kling-Gupta Efficiency (KGE):
      - Gupta, H. V., & Kling, H. (2011). "On typical range, sensitivity, and normalization of Mean Squared Error and
        Nash-Sutcliffe Efficiency type metrics." Water Resources Research, 47(10). https://doi.org/10.1029/2011WR010962
      - Kling, H., Fuchs, M., & Paulin, M. (2012). "Runoff conditions in the upper Danube basin under an ensemble of
        climate change scenarios." Journal of Hydrology, 424-425, 264-277. https://doi.org/10.1016/j.jhydrol.2012.01.011
    """

    class spot_setup_class:
        """
        spot_setup_class: A SPOTPY setup class for MATILDA model parameter calibration.

        This class is designed to facilitate the optimization and calibration of the MATILDA model parameters
        using SPOTPY, a statistical parameter optimization library. It defines the parameter space, simulation
        methods, evaluation functions, and the objective function to assess the model's performance.

        Attributes
        ----------
        param : list
            List of SPOTPY parameter distributions for calibration.
        param_names : list
            List of names corresponding to the parameters being calibrated.
        par_iter : int
            Number of parameter iterations needed for sensitivity analysis and parameter inference.

        Methods
        -------
        __init__(df, obs, swe, obj_func=None)
            Initializes the setup class with input data, observations, and an optional objective function.

        simulation(x, param_names=param_names, fix_param=fix_param, fix_val=fix_val, swe_scaling=swe_scaling)
            Runs the MATILDA simulation with sampled parameters and any fixed parameters specified.

        evaluation()
            Prepares the observed data (e.g., streamflow, snow water equivalent, surface mass balance) for
            comparison with simulation results.

        objectivefunction(simulation, evaluation, params=None)
            Calculates the objective function to evaluate model performance. Supports metrics such as
            the Kling-Gupta Efficiency (KGE) and custom metrics defined by the user.

        Usage
        -----
        The `spot_setup_class` is instantiated with observed data (`obs`) and optionally with snow water equivalent
        (SWE) data and a custom objective function. Once instantiated, it can be passed to SPOTPY's optimization
        and sensitivity analysis tools to calibrate the MATILDA model.

        Example
        -------
        from spotpy.algorithms import sceua
        spot_setup_instance = spot_setup(df, obs, swe, obj_func)
        sampler = sceua(spot_setup_instance)
        sampler.sample(1000)

        References
        ----------
        - SPOTPY library:
          - Houska, T., et al. (2015). "SPOTting Model Parameters Using a Ready-Made Python Package." PLOS ONE.
          - GitHub: https://github.com/thouska/spotpy

        - Kling-Gupta Efficiency (KGE):
          - Gupta, H. V., & Kling, H. (2011). "On typical range, sensitivity, and normalization of Mean Squared Error
            and Nash-Sutcliffe Efficiency type metrics." Water Resources Research.
          - Kling, H., et al. (2012). "Runoff conditions in the upper Danube basin under an ensemble of climate
            change scenarios." Journal of Hydrology.
        """

        # defining all parameters and the distribution
        lr_temp = Uniform(low=lr_temp_lo, high=lr_temp_up)
        lr_prec = Uniform(low=lr_prec_lo, high=lr_prec_up)
        BETA = Uniform(low=BETA_lo, high=BETA_up)
        CET = Uniform(low=CET_lo, high=CET_up)
        FC = Uniform(low=FC_lo, high=FC_up)
        K0 = Uniform(low=K0_lo, high=K0_up)
        K1 = Uniform(low=K1_lo, high=K1_up)
        K2 = Uniform(low=K2_lo, high=K2_up)
        LP = Uniform(low=LP_lo, high=LP_up)
        MAXBAS = Uniform(low=MAXBAS_lo, high=MAXBAS_up)
        PERC = Uniform(low=PERC_lo, high=PERC_up)
        UZL = Uniform(low=UZL_lo, high=UZL_up)
        PCORR = Uniform(low=PCORR_lo, high=PCORR_up)
        TT_snow = Uniform(low=TT_snow_lo, high=TT_snow_up)
        TT_diff = Uniform(low=TT_diff_lo, high=TT_diff_up)
        CFMAX_snow = Uniform(low=CFMAX_snow_lo, high=CFMAX_snow_up)
        CFMAX_rel = Uniform(low=CFMAX_rel_lo, high=CFMAX_rel_up)
        SFCF = Uniform(low=SFCF_lo, high=SFCF_up)
        CWH = Uniform(low=CWH_lo, high=CWH_up)
        AG = Uniform(low=AG_lo, high=AG_up)
        CFR = Uniform(low=CFR_lo, high=CFR_up)

        # Create the list containing the variables
        param = [
            lr_temp,
            lr_prec,
            BETA,
            CET,
            FC,
            K0,
            K1,
            K2,
            LP,
            MAXBAS,
            PERC,
            UZL,
            PCORR,
            TT_snow,
            TT_diff,
            CFMAX_snow,
            CFMAX_rel,
            SFCF,
            CWH,
            AG,
            CFR,
        ]

        # Exclude parameters defined in fix_param
        param_names = [
            "lr_temp",
            "lr_prec",
            "BETA",
            "CET",
            "FC",
            "K0",
            "K1",
            "K2",
            "LP",
            "MAXBAS",
            "PERC",
            "UZL",
            "PCORR",
            "TT_snow",
            "TT_diff",
            "CFMAX_snow",
            "CFMAX_rel",
            "SFCF",
            "CWH",
            "AG",
            "CFR",
        ]

        # Exclude parameters that should be fixed
        if fix_param:
            param = [p for p, name in zip(param, param_names) if name not in fix_param]
            param_names = [param for param in param_names if param not in fix_param]
            for par in fix_param:
                if par in globals():
                    del globals()[par]
                if par in locals():
                    del locals()[par]

        # Number of needed parameter iterations for parametrization and sensitivity analysis
        M = interf  # inference factor (default = 4)
        d = freqst  # frequency step (default = 2)
        k = len(param)  # number of parameters

        par_iter = (1 + 4 * M**2 * (1 + (k - 2) * d)) * k

        def __init__(
            self,
            df,
            obs,
            swe,
            swe_scaling=swe_scaling,
            fix_param=fix_param,
            fix_val=fix_val,
            obj_func=obj_func,
        ):
            self.obj_func = obj_func
            self.Input = df
            self.obs = obs
            self.swe = swe
            self.swe_scaling = swe_scaling
            self.fix_param = fix_param
            self.fix_val = fix_val

        def simulation(self, x, param_names=None):
            if param_names is None:
                param_names = self.param_names

            with HiddenPrints():
                # Setup all parameters for sampling
                args = {par_name: x[par_name] for par_name in param_names}

                # Fix parameters on desired values if defined
                if self.fix_param is not None and self.fix_val is not None:
                    for p in self.fix_param:
                        if p in self.fix_val:
                            args[p] = self.fix_val[p]

                sim = matilda_simulation(
                    self.Input,
                    obs=self.obs,
                    output=None,
                    set_up_start=set_up_start,
                    set_up_end=set_up_end,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    freq=freq,
                    lat=lat,
                    area_cat=area_cat,
                    area_glac=area_glac,
                    ele_dat=ele_dat,
                    ele_glac=ele_glac,
                    ele_cat=ele_cat,
                    plots=False,
                    warn=False,
                    glacier_profile=glacier_profile,
                    elev_rescaling=elev_rescaling,
                    **args,
                )
                swe_sim = (
                    sim[0]
                    .snowpack_off_glaciers["2000-01-01":"2017-09-30"]
                    .to_frame(name="SWE_sim")
                )
                if self.swe_scaling is not None:
                    swe_sim = swe_sim * self.swe_scaling

            if target_mb is None:
                if target_swe is None:
                    return sim[0].total_runoff
                return [sim[0].total_runoff, swe_sim.SWE_sim]

            if target_swe is None:
                return [sim[0].total_runoff, sim[5].smb_water_year.mean()]
            return [
                sim[0].total_runoff,
                sim[5].smb_water_year.mean(),
                swe_sim.SWE_sim,
            ]

        def evaluation(self):
            obs_preproc = self.obs.copy()
            obs_preproc.set_index("Date", inplace=True)
            obs_preproc.index = pd.to_datetime(obs_preproc.index)
            obs_preproc = obs_preproc[sim_start:sim_end]
            # Changing the input unit from m³/s to mm.
            obs_preproc["Qobs"] = (
                obs_preproc["Qobs"] * 86400 / (area_cat * 1000000) * 1000
            )
            # To daily resolution
            obs_preproc = obs_preproc.resample("D").agg(pd.Series.sum, skipna=False)
            # Expanding the observation period full years filling up with NAs
            idx_first = obs_preproc.index.year[1]
            idx_last = obs_preproc.index.year[-1]
            idx = pd.date_range(
                start=date(idx_first, 1, 1),
                end=date(idx_last, 12, 31),
                freq="D",
                name=obs_preproc.index.name,
            )
            obs_preproc = obs_preproc.reindex(idx)
            obs_preproc = obs_preproc.fillna(np.NaN)

            if target_swe is not None:
                swe_obs = self.swe
                if "Date" in self.swe.columns:
                    swe_obs["Date"] = pd.to_datetime(swe_obs["Date"])
                    swe_obs.set_index("Date", inplace=True)
                swe_obs = swe_obs * 1000
                swe_obs = swe_obs["2000-01-01":"2017-09-30"]

            if target_mb is None:
                if target_swe is None:
                    return obs_preproc.Qobs
                return [obs_preproc.Qobs, swe_obs.SWE_Mean]

            if target_swe is None:
                return [obs_preproc.Qobs, target_mb]
            return [obs_preproc.Qobs, target_mb, swe_obs.SWE_Mean]

        def objectivefunction(self, simulation, evaluation):
            # SPOTPY expects to get one or multiple values back,
            # that define the performance of the model run
            if target_mb is not None:
                if target_swe is not None:
                    sim_runoff, sim_smb, sim_swe = simulation
                    eval_runoff, eval_smb, eval_swe = evaluation
                    obj3 = he.kge_2012(sim_swe, eval_swe, remove_zero=False)
                else:
                    sim_runoff, sim_smb = simulation
                    eval_runoff, eval_smb = evaluation

                obj2 = abs(eval_smb - sim_smb)
                simulation = sim_runoff
                evaluation = eval_runoff

            elif target_swe is not None:
                sim_runoff, sim_swe = simulation
                eval_runoff, eval_swe = evaluation
                obj3 = he.kge_2012(sim_swe, eval_swe, remove_zero=False)

            # Crop both timeseries to same periods without NAs
            sim_new = pd.DataFrame()
            sim_new["mod"] = pd.DataFrame(simulation)
            sim_new["obs"] = evaluation
            clean = sim_new.dropna()

            simulation_clean = clean["mod"]
            evaluation_clean = clean["obs"]

            if not self.obj_func:
                # This is used if not overwritten by user
                # obj1 = kge(evaluation_clean, simulation_clean)          # SPOTPY internal kge
                obj1 = he.kge_2012(
                    simulation_clean, evaluation_clean, remove_zero=True
                )  # same as MATILDA
            else:
                # Way to ensure flexible spot setup class
                obj1 = self.obj_func(evaluation_clean, simulation_clean)

            if target_mb is None:
                if target_swe is None:
                    return obj1
                return [obj1, obj3]

            if target_swe is None:
                return [obj1, obj2]

            return [obj1, obj2, obj3]

    return spot_setup_class


def spot_setup_glacier(
    set_up_start=None,
    set_up_end=None,
    sim_start=None,
    sim_end=None,
    freq="D",
    lat=None,
    area_cat=None,
    area_glac=None,
    ele_dat=None,
    ele_glac=None,
    glacier_profile=None,
    obs_type="annual",
    obj_func=None,
    lr_temp_lo=-0.0065,
    lr_temp_up=-0.0055,
    lr_prec_lo=0,
    lr_prec_up=0.002,
    PCORR_lo=0.5,
    PCORR_up=2,
    TT_snow_lo=-1.5,
    TT_snow_up=1.5,
    TT_diff_lo=0.5,
    TT_diff_up=2.5,
    CFMAX_snow_lo=0.5,
    CFMAX_snow_up=10,
    CFMAX_rel_lo=1.2,
    CFMAX_rel_up=2,
    SFCF_lo=0.4,
    SFCF_up=1,
    CFR_lo=0.05,
    CFR_up=0.25,
    interf=4,
    freqst=2,
):
    """
    Spotpy-based setup for parameter optimization and calibration of the MATILDA model for a (sub-)catchment entirely
    covered by glaciers.

    This class provides the framework to optimize the parameters of the MATILDA model using SPOTPY, a statistical parameter
    optimization tool. It enables flexible parameter setup, handles fixed parameters, and evaluates simulation results
    against observed data (e.g., streamflow, snow water equivalent, surface mass balance). It supports annual, winter,
    and summer mass balance optimization based on observed data.

    Parameters
    ----------
    General inputs:
    - set_up_start, set_up_end : str
        Start and end dates of the setup period.
    - sim_start, sim_end : str
        Start and end dates of the simulation period.
    - freq : str, optional
        Frequency of the data, default is daily ("D").
    - lat : float, optional
        Latitude of the catchment for extraterrestrial radiation calculation.
    - area_cat, area_glac : float
        Catchment and glacierized areas in km².
    - ele_dat, ele_glac, ele_cat : float
        Mean elevations (m a.s.l.) of the data, glacier, and catchment.
    - glacier_profile : DataFrame
        Glacier profile used for elevation rescaling and look-up table generation.
    - obs_type : str, optional
        Type of observed mass balance data: "annual", "winter", or "summer". Default is "annual".
    - obj_func : callable, optional
        Custom objective function to compare simulation and observation. If None, the Mean Absolute Error (MAE) is used.

    Parameter bounds for calibration:
    - lr_temp_lo, lr_temp_up : float
        Bounds for temperature lapse rate (°C/m).
    - lr_prec_lo, lr_prec_up : float
        Bounds for precipitation correction factor.
    - PCORR_lo, PCORR_up : float
        Bounds for precipitation correction factor.
    - TT_snow_lo, TT_snow_up : float
        Bounds for snow temperature threshold (°C).
    - TT_diff_lo, TT_diff_up : float
        Bounds for the difference between rain and snow temperature thresholds (°C).
    - CFMAX_snow_lo, CFMAX_snow_up : float
        Bounds for the degree-day factor for snow melt (mm/°C/day).
    - CFMAX_rel_lo, CFMAX_rel_up : float
        Bounds for the degree-day factor for relative snow and ice melt.
    - SFCF_lo, SFCF_up : float
        Bounds for snow correction factor.
    - CFR_lo, CFR_up : float
        Bounds for the refreezing factor.

    SPOTPY settings:
    - interf : int, optional
        Inference factor for parameter iterations, default is 4.
    - freqst : int, optional
        Frequency step for sensitivity analysis, default is 2.

    Methods
    -------
    simulation(x)
        Executes the glacier mass balance simulation using the degree-day model (DDM).
    evaluation()
        Prepares observed mass balance data (annual, winter, or summer) for comparison.
    objectivefunction(simulation, evaluation, params=None)
        Defines the objective function to calculate the error between simulated and observed mass balance values.

    Returns
    -------
    spot_setup_class : class
        Configured SPOTPY setup class for glacier mass balance parameter optimization.

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy
    """

    class spot_setup_class:
        """
        spot_setup_class: A SPOTPY setup class for MATILDA model glacier mass balance parameter calibration.

        This class facilitates the calibration and optimization of the MATILDA model parameters for glacier mass balance
        modeling. It uses SPOTPY, a statistical parameter optimization library, to define the parameter space, simulate
        mass balance, and evaluate the model's performance against observed data.

        Attributes
        ----------
        param : list
            List of SPOTPY parameter distributions for calibration, including lapse rates, temperature thresholds, and
            degree-day factors.
        par_iter : int
            Number of parameter iterations needed for sensitivity analysis and parameter inference.

        Methods
        -------
        __init__(df, obs, obj_func=None)
            Initializes the class with input data (`df`), observed data (`obs`), and an optional custom objective function.

        simulation(x)
            Simulates glacier mass balance using the MATILDA model with the specified parameters.

        evaluation()
            Processes the observed mass balance data into a format compatible with the simulated outputs.

        objectivefunction(simulation, evaluation, params=None)
            Defines the objective function to evaluate the model's performance. By default, it uses the Mean Absolute
            Error (MAE) or a custom user-defined function.

        Usage
        -----
        The `spot_setup_class` is instantiated with observed data and passed to SPOTPY for parameter optimization and
        sensitivity analysis.

        Example
        -------
        from spotpy.algorithms import sceua

        spot_setup_instance = spot_setup_glacier(df, obs, obj_func)
        sampler = sceua(spot_setup_instance)
        sampler.sample(1000)

        Parameters
        ----------
        General Inputs:
        - set_up_start, set_up_end : str
            Start and end dates of the setup period.
        - sim_start, sim_end : str
            Start and end dates of the simulation period.
        - freq : str, optional
            Frequency of the data, default is daily ("D").
        - lat : float, optional
            Latitude of the catchment for extraterrestrial radiation calculation.
        - area_cat, area_glac : float
            Catchment and glacierized areas in km².
        - ele_dat, ele_glac, ele_cat : float
            Mean elevations (m a.s.l.) of the data, glacier, and catchment.
        - glacier_profile : DataFrame
            Glacier profile used for elevation rescaling and look-up table generation.
        - obs_type : str, optional
            Type of observed mass balance data ("annual", "winter", or "summer"). Default is "annual".
        - obj_func : callable, optional
            Custom objective function to compare simulation and observation. Defaults to Mean Absolute Error (MAE).

        Parameter Bounds for Calibration:
        - lr_temp_lo, lr_temp_up : float
            Bounds for the temperature lapse rate (°C/m).
        - lr_prec_lo, lr_prec_up : float
            Bounds for the precipitation correction factor.
        - PCORR_lo, PCORR_up : float
            Bounds for the precipitation correction factor.
        - TT_snow_lo, TT_snow_up : float
            Bounds for the snow temperature threshold (°C).
        - TT_diff_lo, TT_diff_up : float
            Bounds for the difference between rain and snow temperature thresholds (°C).
        - CFMAX_snow_lo, CFMAX_snow_up : float
            Bounds for the degree-day factor for snow melt (mm/°C/day).
        - CFMAX_rel_lo, CFMAX_rel_up : float
            Bounds for the degree-day factor for relative snow and ice melt.
        - SFCF_lo, SFCF_up : float
            Bounds for the snow correction factor.
        - CFR_lo, CFR_up : float
            Bounds for the refreezing factor.

        SPOTPY Settings:
        - interf : int, optional
            Inference factor for parameter iterations. Default is 4.
        - freqst : int, optional
            Frequency step for sensitivity analysis. Default is 2.

        Returns
        -------
        spot_setup_class : class
            Configured SPOTPY setup class for glacier mass balance parameter optimization.

        References
        ----------
        - SPOTPY library:
          - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
            Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
          - GitHub repository: https://github.com/thouska/spotpy
        """

        # defining all parameters and the distribution
        param = (
            lr_temp,
            lr_prec,
            PCORR,
            TT_snow,
            TT_diff,
            CFMAX_snow,
            CFMAX_rel,
            SFCF,
            CFR,
        ) = [
            Uniform(low=lr_temp_lo, high=lr_temp_up),  # lr_temp
            Uniform(low=lr_prec_lo, high=lr_prec_up),  # lr_prec
            Uniform(low=PCORR_lo, high=PCORR_up),  # PCORR
            Uniform(low=TT_snow_lo, high=TT_snow_up),  # TT_snow
            Uniform(low=TT_diff_lo, high=TT_diff_up),  # TT_diff
            Uniform(low=CFMAX_snow_lo, high=CFMAX_snow_up),  # CFMAX_snow
            Uniform(low=CFMAX_rel_lo, high=CFMAX_rel_up),  # CFMAX_rel
            Uniform(low=SFCF_lo, high=SFCF_up),  # SFCF
            Uniform(low=CFR_lo, high=CFR_up),  # CFR
        ]

        # Number of needed parameter iterations for parametrization and sensitivity analysis
        M = interf  # inference factor (default = 4)
        d = freqst  # frequency step (default = 2)
        k = len(param)  # number of parameters

        par_iter = (1 + 4 * M**2 * (1 + (k - 2) * d)) * k

        def __init__(self, df, obs, obj_func=obj_func):
            self.obj_func = obj_func
            self.Input = df
            self.obs = obs

        def simulation(self, x):
            with HiddenPrints():
                parameter = matilda_parameter(
                    self.Input,
                    set_up_start=set_up_start,
                    set_up_end=set_up_end,
                    sim_start=sim_start,
                    sim_end=sim_end,
                    freq=freq,
                    lat=lat,
                    area_cat=area_cat,
                    area_glac=area_glac,
                    ele_dat=ele_dat,
                    ele_glac=ele_glac,
                    ele_cat=None,
                    lr_temp=x.lr_temp,
                    lr_prec=x.lr_prec,
                    PCORR=x.PCORR,
                    TT_snow=x.TT_snow,
                    TT_diff=x.TT_diff,
                    CFMAX_snow=x.CFMAX_snow,
                    CFMAX_rel=x.CFMAX_rel,
                    SFCF=x.SFCF,
                    CFR=x.CFR,
                )

                df_preproc = matilda_preproc(self.Input, parameter)
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM = updated_glacier_melt(
                    df_preproc, lookup_table, glacier_profile, parameter
                )[0]
                sim = output_DDM.DDM_smb

            return sim

        def evaluation(self):
            obs_preproc = self.obs.copy()
            obs_preproc.set_index("YEAR", inplace=True)
            obs_preproc.index = pd.to_datetime(obs_preproc.index)

            return obs_preproc

        def objectivefunction(self, simulation, evaluation):
            # Aggregate MBs to fit the calibration data
            sim = []
            obs = []
            sim_new = pd.DataFrame()
            if obs_type == "winter":
                for i in evaluation.index:
                    mb = simulation[winter(i, evaluation)].sum()
                    sim.append(mb)
                    mb_obs = evaluation[evaluation.index == i].WINTER_BALANCE.squeeze()
                    obs.append(mb_obs)

            if obs_type == "summer":
                for i in evaluation.index:
                    mb = simulation[summer(i, evaluation)].sum()
                    sim.append(mb)
                    mb_obs = evaluation[evaluation.index == i].SUMMER_BALANCE.squeeze()
                    obs.append(mb_obs)

            if obs_type == "annual":
                for i in evaluation.index:
                    mb_sim = simulation[annual(i, evaluation)].sum()
                    sim.append(mb_sim)
                    mb_obs = evaluation[evaluation.index == i].ANNUAL_BALANCE.squeeze()
                    obs.append(mb_obs)

            # Crop both timeseries to same periods without NAs
            sim_new["mod"] = pd.DataFrame(sim)
            sim_new["obs"] = pd.DataFrame(obs)

            clean = sim_new.dropna()

            simulation_clean = clean["mod"]
            evaluation_clean = clean["obs"]

            if not self.obj_func:
                # This is used if not overwritten by user
                like = mae(evaluation_clean, simulation_clean)
                print("MAE: " + str(like))
            else:
                # Way to ensure flexible spot setup class
                like = self.obj_func(evaluation_clean, simulation_clean)
            return like

    return spot_setup_class


def analyze_results(
    sampling_data,
    obs,
    algorithm,
    obj_dir="maximize",
    fig_path=None,
    dbname="mspot_results",
    glacier_only=False,
    target_mb=None,
):
    """
    Analyze the results of SPOTPY sampling for model calibration of the MATILDA model.

    This function processes sampling results from SPOTPY to identify the best parameter set, evaluate performance,
    and optionally generate visualizations of the results.

    Parameters
    ----------
    sampling_data : str or SPOTPY result object
        The file path to a SPOTPY results CSV file or a SPOTPY result object.
    obs : str or pandas.DataFrame
        The file path to observed data (in CSV format with 'Date' column) or a DataFrame containing observed data.
    algorithm : str
        The SPOTPY algorithm used for sampling (e.g., 'abc', 'dream', 'sceua').
    obj_dir : str, optional
        Direction of the objective function optimization, either 'maximize' or 'minimize'. Default is 'maximize'.
    fig_path : str, optional
        Path to save generated plots. If None, plots are not saved.
    dbname : str, optional
        Name for saving the database and plots. Default is 'mspot_results'.
    glacier_only : bool, optional
        If True, optimizes only for glacier-specific metrics (e.g., mass balance). Default is False.
    target_mb : float, optional
        Target mean annual mass balance for melt model calibration. If None, this metric is not considered. Default is None.

    Returns
    -------
    dict
        A dictionary containing:
        - 'best_param': dict
            Best parameter set identified during sampling.
        - 'best_index': int
            Index of the best run in the SPOTPY results.
        - 'best_model_run': array
            Model output corresponding to the best parameter set.
        - 'best_objf': float
            Value of the objective function for the best parameter set.
        - 'best_simulation': pandas.Series (if glacier_only is False and target_mb is None)
            Simulated time series corresponding to the best parameter set.
        - 'sampling_plot': matplotlib.Figure (if applicable)
            Iteration vs. objective function plot.
        - 'best_run_plot': matplotlib.Figure (if applicable)
            Best simulation vs. observed data plot.
        - 'par_uncertain_plot': matplotlib.Figure (if applicable)
            Parameter uncertainty plot.

    Notes
    -----
    - The objective function can be maximized (e.g., Kling-Gupta Efficiency) or minimized (e.g., Mean Absolute Error)
      based on the algorithm and obj_dir setting.
    - For glacier-specific calibration, only glacier mass balance data is considered.

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy
    """

    if isinstance(obs, str):
        if glacier_only:
            obs = pd.read_csv(obs)
        else:
            obs = pd.read_csv(obs, index_col="Date", parse_dates=["Date"])

    if isinstance(sampling_data, str):
        if sampling_data.endswith(".csv"):
            sampling_data = sampling_data[: len(sampling_data) - 4]
        results = spotpy.analyser.load_csv_results(sampling_data)
    else:
        results = sampling_data.getdata()

    maximize = ["abc", "dds", "demcz", "dream", "rope", "sa", "fscabc", "mcmc", "mle"]
    minimize = ["nsgaii", "padds", "sceua"]
    both = ["fast", "lhs", "mc"]

    if algorithm in maximize:
        best_param = spotpy.analyser.get_best_parameterset(results)
    elif algorithm in minimize:
        best_param = spotpy.analyser.get_best_parameterset(results, maximize=False)
    elif algorithm in both:
        print(
            "WARNING: The selected algorithm "
            + algorithm
            + " can either maximize or minimize the objective function."
            " You can specify the direction by passing obj_dir to analyze_results(). The default is 'maximize'."
        )
        if obj_dir == "maximize":
            best_param = spotpy.analyser.get_best_parameterset(results)
        elif obj_dir == "minimize":
            best_param = spotpy.analyser.get_best_parameterset(results, maximize=False)
        else:
            raise ValueError(
                "Invalid argument for obj_dir. Choose 'minimize' or 'maximize'."
            )
    else:
        raise ValueError(
            "Invalid argument for algorithm. Available algorithms: ['abc', 'dds', 'demcz', 'dream', 'rope', 'sa',"
            "'fscabc', 'mcmc', 'mle', 'nsgaii', 'padds', 'sceua', 'fast', 'lhs', 'mc']"
        )

    par_names = spotpy.analyser.get_parameternames(best_param)
    param_zip = zip(par_names, best_param[0])
    best_param = dict(param_zip)

    if glacier_only:
        bestindex, bestobjf = spotpy.analyser.get_minlikeindex(
            results
        )  # Run with lowest MAE
        best_model_run = results[bestindex]

    elif target_mb is not None:
        if obj_dir == "minimize":
            bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
        else:
            bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)
        best_model_run = results[bestindex]

    else:
        sim_start = obs.index[0]
        sim_end = obs.index[-1]
        bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(
            results
        )  # Run with highest KGE
        best_model_run = results[bestindex]
        fields = [word for word in best_model_run.dtype.names if word.startswith("sim")]
        best_simulation = pd.Series(
            list(list(best_model_run[fields])[0]),
            index=pd.date_range(sim_start, sim_end),
        )
        # Only necessary because obs has a datetime. Thus, both need a datetime.

    if not glacier_only and target_mb is None:
        fig1 = plt.figure(1, figsize=(9, 5))
        plt.plot(results["like1"])
        plt.ylabel("KGE")
        plt.xlabel("Iteration")
        if fig_path is not None:
            plt.savefig(fig_path + "/" + dbname + "_sampling_plot.png")

        fig2 = plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 1, 1)
        ax.plot(
            best_simulation,
            color="black",
            linestyle="solid",
            label="Best objf.=" + str(bestobjf),
        )
        ax.plot(obs, "r.", markersize=3, label="Observation data")
        plt.xlabel("Date")
        plt.ylabel("Discharge [mm d-1]")
        plt.legend(loc="upper right")
        if fig_path is not None:
            plt.savefig(fig_path + "/" + dbname + "_best_run_plot.png")

        fig3 = plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 1, 1)
        q5, q95 = [], []
        for field in fields:
            q5.append(np.percentile(results[field][-100:-1], 2.5))
            q95.append(np.percentile(results[field][-100:-1], 97.5))
        ax.plot(q5, color="dimgrey", linestyle="solid")
        ax.plot(q95, color="dimgrey", linestyle="solid")
        ax.fill_between(
            np.arange(0, len(q5), 1),
            list(q5),
            list(q95),
            facecolor="dimgrey",
            zorder=0,
            linewidth=0,
            label="parameter uncertainty",
        )
        ax.plot(
            np.array(obs), "r.", label="data"
        )  # Need to remove Timestamp from Evaluation to make comparable
        # ax.set_ylim(0, 100)
        ax.set_xlim(0, len(obs))
        ax.legend()
        if fig_path is not None:
            plt.savefig(fig_path + "/" + dbname + "_par_uncertain_plot.png")

        return {
            "best_param": best_param,
            "best_index": bestindex,
            "best_model_run": best_model_run,
            "best_objf": bestobjf,
            "best_simulation": best_simulation,
            "sampling_plot": fig1,
            "best_run_plot": fig2,
            "par_uncertain_plot": fig3,
        }

    return {
        "best_param": best_param,
        "best_index": bestindex,
        "best_model_run": best_model_run,
        "best_objf": bestobjf,
    }


def psample(
    df,
    obs,
    rep=10,
    output=None,
    dbname="matilda_par_smpl",
    dbformat=None,
    obj_func=None,
    opt_iter=False,
    fig_path=None,  # savefig=False,
    set_up_start=None,
    set_up_end=None,
    sim_start=None,
    sim_end=None,
    freq="D",
    lat=None,
    area_cat=None,
    area_glac=None,
    ele_dat=None,
    ele_glac=None,
    ele_cat=None,
    glacier_profile=None,
    interf=4,
    freqst=2,
    parallel=False,
    cores=2,
    save_sim=True,
    elev_rescaling=True,
    glacier_only=False,
    obs_type="annual",
    target_mb=None,
    target_swe=None,
    swe_scaling=None,
    algorithm="lhs",
    obj_dir="maximize",
    fix_param=None,
    fix_val=None,
    demcz_args: dict = None,  # DEMCz settings
    **kwargs,
):
    """
    Run parameter sampling for the MATILDA model using SPOTPY.

    This function sets up a SPOTPY sampling routine for parameter optimization or sensitivity analysis.
    It supports parallel and serial computation, custom objective functions, and a variety of SPOTPY algorithms.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data for the MATILDA model.
    obs : pandas.DataFrame or str
        Observed data as a DataFrame or a file path to a CSV file containing observations.
    rep : int, optional
        Number of sampling repetitions (default is 10).
    output : str, optional
        Path to save the output files (default is None).
    dbname : str, optional
        Name of the database for saving results (default is 'matilda_par_smpl').
    dbformat : str, optional
        Format for the SPOTPY database (e.g., 'csv', 'sql'). Default is None (no database saved).
    obj_func : callable, optional
        Custom objective function for parameter optimization. If None, defaults are used.
    opt_iter : bool, optional
        If True, samples the optimum number of iterations based on the setup (default is False).
    fig_path : str, optional
        Path to save generated plots (default is None).
    set_up_start, set_up_end : str, optional
        Start and end dates for the model setup period (e.g., '2000-01-01').
    sim_start, sim_end : str, optional
        Start and end dates for the simulation period.
    freq : str, optional
        Temporal resolution of the input data (default is 'D').
    lat : float, optional
        Latitude of the study area (used for potential evapotranspiration calculations).
    area_cat : float, optional
        Total catchment area in km².
    area_glac : float, optional
        Glacierized area in km².
    ele_dat, ele_glac, ele_cat : float, optional
        Elevations (mean, glacier, and catchment).
    glacier_profile : pandas.DataFrame, optional
        Glacier profile data for rescaling and optimization routines.
    interf : int, optional
        Inference factor for SPOTPY (default is 4).
    freqst : int, optional
        Frequency step for SPOTPY (default is 2).
    parallel : bool, optional
        If True, runs sampling in parallel using MPI (default is False).
    cores : int, optional
        Number of cores to use for parallel sampling (default is 2).
    save_sim : bool, optional
        Whether to save simulation results in the database (default is True).
    elev_rescaling : bool, optional
        If True, applies elevation-based rescaling for glacier simulations (default is True).
    glacier_only : bool, optional
        If True, runs sampling specifically for glacier-related parameters (default is False).
    obs_type : str, optional
        Type of observed data for glacier calibration ('annual', 'winter', or 'summer'). Default is 'annual'.
    target_mb : float, optional
        Target mass balance for glacier calibration (default is None).
    target_swe : float, optional
        Target snow water equivalent for calibration (default is None).
    swe_scaling : float, optional
        Scaling factor for snow water equivalent (default is None).
    algorithm : str, optional
        SPOTPY algorithm to use (e.g., 'lhs', 'mcmc', 'dream'). Default is 'lhs'.
    obj_dir : str, optional
        Direction of the objective function optimization ('maximize' or 'minimize'). Default is 'maximize'.
    fix_param : list of str, optional
        List of parameters to fix during sampling (default is None).
    fix_val : dict, optional
        Dictionary of fixed parameter values (default is None).
    demcz_args : dict, optional
        Additional arguments for the DEMCz algorithm (default is None).
    kwargs : dict, optional
        Additional parameters for the MATILDA model.

    Returns
    -------
    dict
        Results of the sampling, including:
        - 'best_param': dict
            Best parameter set identified during sampling.
        - 'best_index': int
            Index of the best run in the SPOTPY results.
        - 'best_model_run': array
            Model output corresponding to the best parameter set.
        - 'best_objf': float
            Value of the objective function for the best parameter set.
        - 'best_simulation': pandas.Series (if applicable)
            Simulated time series corresponding to the best parameter set.
        - 'sampling_plot': matplotlib.Figure (if applicable)
            Iteration vs. objective function plot.
        - 'best_run_plot': matplotlib.Figure (if applicable)
            Best simulation vs. observed data plot.
        - 'par_uncertain_plot': matplotlib.Figure (if applicable)
            Parameter uncertainty plot.

    Notes
    -----
    - This function uses the SPOTPY library for parameter sampling and optimization.
    - Parallel computation is supported for specific algorithms ('mc', 'lhs', 'fast', 'rope', 'sceua', 'demcz').

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy
    """

    cwd = os.getcwd()
    if output is not None:
        os.chdir(output)

    # Setup model class:

    if glacier_only:
        setup = spot_setup_glacier(
            set_up_start=set_up_start,
            set_up_end=set_up_end,
            sim_start=sim_start,
            sim_end=sim_end,
            freq=freq,
            area_cat=area_cat,
            area_glac=area_glac,
            ele_dat=ele_dat,
            ele_glac=ele_glac,
            lat=lat,
            interf=interf,
            freqst=freqst,
            glacier_profile=glacier_profile,
            obs_type=obs_type,
            **kwargs,
        )

    else:
        setup = spot_setup(
            set_up_start=set_up_start,
            set_up_end=set_up_end,
            sim_start=sim_start,
            sim_end=sim_end,
            freq=freq,
            area_cat=area_cat,
            area_glac=area_glac,
            ele_dat=ele_dat,
            ele_glac=ele_glac,
            ele_cat=ele_cat,
            lat=lat,
            interf=interf,
            freqst=freqst,
            glacier_profile=glacier_profile,
            elev_rescaling=elev_rescaling,
            target_mb=target_mb,
            fix_param=fix_param,
            fix_val=fix_val,
            target_swe=target_swe,
            swe_scaling=swe_scaling,
            **kwargs,
        )

    psample_setup = setup(
        df, obs, target_swe, obj_func
    )  # Define custom objective function using obj_func=
    alg_selector = {
        "mc": spotpy.algorithms.mc,
        "sceua": spotpy.algorithms.sceua,
        "mcmc": spotpy.algorithms.mcmc,
        "mle": spotpy.algorithms.mle,
        "abc": spotpy.algorithms.abc,
        "sa": spotpy.algorithms.sa,
        "dds": spotpy.algorithms.dds,
        "demcz": spotpy.algorithms.demcz,
        "dream": spotpy.algorithms.dream,
        "fscabc": spotpy.algorithms.fscabc,
        "lhs": spotpy.algorithms.lhs,
        "padds": spotpy.algorithms.padds,
        "rope": spotpy.algorithms.rope,
        "fast": spotpy.algorithms.fast,
        "nsgaii": spotpy.algorithms.nsgaii,
    }

    if target_mb is not None:  # Format errors in database csv when saving simulations
        save_sim = False

    if parallel:
        sampler = alg_selector[algorithm](
            psample_setup,
            dbname=dbname,
            dbformat=dbformat,
            parallel="mpi",
            optimization_direction=obj_dir,
            save_sim=save_sim,
        )
        if algorithm in ("mc", "lhs", "fast", "rope"):
            sampler.sample(rep)
        elif algorithm == "sceua":
            sampler.sample(rep, ngs=cores)
        elif algorithm == "demcz":
            sampler.sample(rep, nChains=cores, **demcz_args)
        else:
            raise ValueError(
                "The selected algorithm is ineligible for parallel computing."
                'Either select a different algorithm (mc, lhs, fast, rope, sceua or demcz) or set "parallel = False".'
            )
    else:
        sampler = alg_selector[algorithm](
            psample_setup,
            dbname=dbname,
            dbformat=dbformat,
            save_sim=save_sim,
            optimization_direction=obj_dir,
        )
        if opt_iter:
            if yesno(
                f"\n******** WARNING! Your optimum # of iterations is {psample_setup.par_iter}. "
                "This may take a long time.\n******** Do you wish to proceed"
            ):
                sampler.sample(
                    psample_setup.par_iter
                )  # ideal number of reps = psample_setup.par_iter
            else:
                print(f"Proceeding with {rep} iterations.")
                sampler.sample(rep)
        else:
            sampler.sample(rep)

    # Change dbformat to None for short tests but to 'csv' or 'sql' to avoid data loss in case off long calculations.

    if target_mb is None:
        psample_setup.evaluation().to_csv(dbname + "_observations.csv")
    else:
        psample_setup.evaluation()[0].to_csv(dbname + "_observations.csv")

    if fix_param is not None:
        print("Fixed parameters:\n")
        defaults = {
            "lr_temp": -0.006,
            "lr_prec": 0,
            "TT_snow": 0,
            "TT_diff": 2,
            "CFMAX_snow": 2.5,
            "CFMAX_rel": 2,
            "BETA": 1.0,
            "CET": 0.15,
            "FC": 250,
            "K0": 0.055,
            "K1": 0.055,
            "K2": 0.04,
            "LP": 0.7,
            "MAXBAS": 3.0,
            "PERC": 1.5,
            "UZL": 120,
            "PCORR": 1.0,
            "SFCF": 0.7,
            "CWH": 0.1,
            "AG": 0.7,
            "CFR": 0.15,
        }
        print_par = {}
        for p in fix_param:
            if p in fix_val:
                print_par[p] = fix_val[p]
            else:
                print_par[p] = defaults[p]
        for key, value in print_par.items():
            print(f"{key}: {value}")
        print("\nNOTE: Fixed parameters are not listed in the final parameter set.\n")

    if not parallel:

        if target_mb is None:
            results = analyze_results(
                sampler,
                psample_setup.evaluation(),
                algorithm=algorithm,
                obj_dir=obj_dir,
                fig_path=fig_path,
                dbname=dbname,
                glacier_only=glacier_only,
            )
        else:
            results = analyze_results(
                sampler,
                psample_setup.evaluation()[0],
                algorithm=algorithm,
                obj_dir=obj_dir,
                fig_path=fig_path,
                dbname=dbname,
                glacier_only=glacier_only,
                target_mb=target_mb,
            )
	
	os.chdir(cwd)
	
        return results

    os.chdir(cwd)


def load_parameters(path, algorithm, obj_dir="maximize", glacier_only=False):
    """
    Load the best parameter set from SPOTPY sampling results.

    This function retrieves the best parameter set from a SPOTPY sampling results file, based on the specified algorithm
    and objective function direction.

    Parameters
    ----------
    path : str
        File path to the SPOTPY sampling results (e.g., 'results.csv').
    algorithm : str
        SPOTPY algorithm used for sampling (e.g., 'lhs', 'mcmc', 'dream').
    obj_dir : str, optional
        Direction of the objective function optimization ('maximize' or 'minimize'). Default is 'maximize'.
    glacier_only : bool, optional
        If True, processes only glacier-specific sampling results (default is False).

    Returns
    -------
    dict
        A dictionary containing the best parameter set identified from the sampling results.

    Notes
    -----
    - This function uses the `analyze_results` function to extract the best parameters.
    - It assumes that the corresponding observation file is named `<path_without_extension>_observations.csv`.
    - Depending on the number of samples loading of the sampling file might be slow.

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy
    """

    sampling_csv = path
    sampling_obs = path.split(".")[0] + "_observations.csv"
    results = analyze_results(
        sampling_csv, sampling_obs, algorithm, obj_dir, glacier_only=glacier_only
    )

    return results["best_param"]


def get_par_bounds(path, threshold=10, percentage=True, drop=None):
    """
    Generate parameter bounds from SPOTPY sampling results.

    This function extracts the minimum and maximum values of model parameters from the best-performing runs in
    SPOTPY sampling results. The best-performing runs are determined based on a likelihood threshold or percentage.

    Parameters
    ----------
    path : str
        File path to the SPOTPY sampling results (e.g., 'results.csv').
    threshold : float, optional
        The threshold for selecting the best-performing runs:
        - If `percentage` is True, `threshold` represents the percentage of the top-performing runs to consider.
        - If `percentage` is False, `threshold` represents the numerical threshold for the objective function.
        Default is 10.
    percentage : bool, optional
        Determines whether to interpret `threshold` as a percentage (True) or a numerical threshold (False).
        Default is True.
    drop : list of str, optional
        A list of parameter names to exclude from the resulting bounds. Default is None.

    Returns
    -------
    dict
        A dictionary containing the lower (`_lo`) and upper (`_up`) bounds for each parameter, derived from the
        best-performing runs.

    Notes
    -----
    - The function loads SPOTPY sampling results and identifies the best-performing runs using the specified threshold
      and criteria.
    - Parameters specified in the `drop` list are excluded from the output bounds.
    - The resulting dictionary contains keys in the format `<parameter_name>_lo` and `<parameter_name>_up`.

    References
    ----------
    - SPOTPY library:
      - Houska, T., Kraft, P., Chamorro-Chavez, A., & Breuer, L. (2015). "SPOTting Model Parameters Using a Ready-Made
        Python Package." PLOS ONE, 10(12), 1-22. https://doi.org/10.1371/journal.pone.0145180
      - GitHub repository: https://github.com/thouska/spotpy
    """
    if drop is None:
        drop = []  # Initialize to an empty list if not provided

    result_path = path
    results = spotpy.analyser.load_csv_results(result_path)
    # Get parameters of best model runs
    if percentage:
        best = spotpy.analyser.get_posterior(
            results, maximize=False, percentage=threshold
        )  # get best xx% runs
    else:
        best = results[
            np.where(results["like1"] <= threshold)
        ]  # get all runs below a treshhold
    params = spotpy.analyser.get_parameters(best)
    # Write min and max parameter values of best model runs to dictionary
    par_bounds = {}
    for i in spotpy.analyser.get_parameter_fields(best):
        p = params[i]
        par_bounds[i.split("par")[1] + "_lo"] = min(p)
        par_bounds[i.split("par")[1] + "_up"] = max(p)
    for i in drop:
        del par_bounds[i + "_lo"]
        del par_bounds[i + "_up"]

    return par_bounds
