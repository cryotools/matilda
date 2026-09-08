"""Compatibility checks for the main public call signatures."""

from __future__ import annotations

import inspect

from matilda.core import matilda_parameter, matilda_preproc, matilda_simulation
from matilda.mspot_glacier import psample, spot_setup, spot_setup_glacier


PUBLIC_SIGNATURES = {
    matilda_parameter: (
        "(input_df, set_up_start=None, set_up_end=None, sim_start=None, "
        "sim_end=None, freq='D', lat=None, area_cat=None, area_glac=None, "
        "ele_dat=None, ele_glac=None, ele_cat=None, warn=False, **matilda_param)"
    ),
    matilda_preproc: "(input_df, parameter, obs=None)",
    matilda_simulation: (
        "(input_df, obs=None, glacier_profile=None, output=None, warn=False, "
        "set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, "
        "freq='D', lat=None, area_cat=None, area_glac=None, ele_dat=None, "
        "ele_glac=None, ele_cat=None, plots=True, plot_type='print', "
        "science_plot=True, elev_rescaling=False, drop_surplus=False, "
        "**matilda_param)"
    ),
    spot_setup: (
        "(set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, "
        "freq='D', lat=None, area_cat=None, area_glac=None, ele_dat=None, "
        "ele_glac=None, ele_cat=None, glacier_profile=None, "
        "elev_rescaling=True, target_mb=None, target_swe=None, "
        "swe_scaling=None, fix_param=None, fix_val=None, obj_func=None, "
        "lr_temp_lo=-0.0065, lr_temp_up=-0.0055, lr_prec_lo=0, "
        "lr_prec_up=0.002, BETA_lo=1, BETA_up=6, CET_lo=0, CET_up=0.3, "
        "FC_lo=50, FC_up=500, K0_lo=0.01, K0_up=0.4, K1_lo=0.01, "
        "K1_up=0.4, K2_lo=0.001, K2_up=0.15, LP_lo=0.3, LP_up=1, "
        "MAXBAS_lo=2, MAXBAS_up=7, PERC_lo=0, PERC_up=3, UZL_lo=0, "
        "UZL_up=500, PCORR_lo=0.5, PCORR_up=2, TT_snow_lo=-1.5, "
        "TT_snow_up=1.5, TT_diff_lo=0.5, TT_diff_up=2.5, "
        "CFMAX_snow_lo=0.5, CFMAX_snow_up=10, CFMAX_rel_lo=1.2, "
        "CFMAX_rel_up=2, SFCF_lo=0.4, SFCF_up=1, CWH_lo=0, CWH_up=0.2, "
        "AG_lo=0, AG_up=1, CFR_lo=0.05, CFR_up=0.25, interf=4, freqst=2)"
    ),
    spot_setup_glacier: (
        "(set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, "
        "freq='D', lat=None, area_cat=None, area_glac=None, ele_dat=None, "
        "ele_glac=None, glacier_profile=None, obs_type='annual', "
        "obj_func=None, lr_temp_lo=-0.0065, lr_temp_up=-0.0055, "
        "lr_prec_lo=0, lr_prec_up=0.002, PCORR_lo=0.5, PCORR_up=2, "
        "TT_snow_lo=-1.5, TT_snow_up=1.5, TT_diff_lo=0.5, "
        "TT_diff_up=2.5, CFMAX_snow_lo=0.5, CFMAX_snow_up=10, "
        "CFMAX_rel_lo=1.2, CFMAX_rel_up=2, SFCF_lo=0.4, SFCF_up=1, "
        "CFR_lo=0.05, CFR_up=0.25, interf=4, freqst=2)"
    ),
    psample: (
        "(df, obs, rep=10, output=None, dbname='matilda_par_smpl', "
        "dbformat=None, obj_func=None, opt_iter=False, fig_path=None, "
        "set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, "
        "freq='D', lat=None, area_cat=None, area_glac=None, ele_dat=None, "
        "ele_glac=None, ele_cat=None, glacier_profile=None, interf=4, "
        "freqst=2, parallel=False, cores=2, save_sim=True, "
        "elev_rescaling=True, glacier_only=False, obs_type='annual', "
        "target_mb=None, target_swe=None, swe_scaling=None, algorithm='lhs', "
        "obj_dir='maximize', fix_param=None, fix_val=None, "
        "demcz_args: dict = None, **kwargs)"
    ),
}


def test_main_public_call_signatures_are_stable():
    actual = {function: str(inspect.signature(function)) for function in PUBLIC_SIGNATURES}

    assert actual == PUBLIC_SIGNATURES
