"""Deterministic synthetic inputs for model-mode and calibration tests."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


SETUP_START = "1999-01-01"
SETUP_END = "1999-12-31"
SIMULATION_START = "2000-01-01"
SIMULATION_END = "2001-12-31"
AREA_CATCHMENT = 100.0
REFERENCE_DIRECTORY = Path(__file__).parent / "test_input"
REFERENCE_MANIFEST_PATH = REFERENCE_DIRECTORY / "synthetic_reference_manifest.json"
SYNTHETIC_FRAME_OUTPUTS = {
    0: "compact_daily",
    1: "full_daily",
    3: "summary_statistics",
}


def make_synthetic_forcing() -> pd.DataFrame:
    """Return seasonal daily forcing containing a leap-year simulation period."""
    dates = pd.date_range(SETUP_START, SIMULATION_END, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    seasonal_angle = 2 * np.pi * (day_of_year - 80) / 365.25

    return pd.DataFrame(
        {
            "TIMESTAMP": dates,
            "T2": -2.0 + 10.0 * np.sin(seasonal_angle),
            "RRR": np.where(np.arange(len(dates)) % 7 == 0, 5.0, 0.0),
            "PE": np.clip(
                1.5
                + np.sin(2 * np.pi * (day_of_year - 100) / 365.25),
                0,
                None,
            ),
        }
    )


def model_settings(area_glac: float) -> dict:
    """Return common public-API settings for a synthetic MATILDA run."""
    return {
        "set_up_start": SETUP_START,
        "set_up_end": SETUP_END,
        "sim_start": SIMULATION_START,
        "sim_end": SIMULATION_END,
        "freq": "D",
        "lat": 45.0,
        "area_cat": AREA_CATCHMENT,
        "area_glac": area_glac,
        "ele_dat": 1000.0,
        "ele_cat": 1200.0,
        "ele_glac": 1600.0,
        "plots": False,
        "elev_rescaling": False,
        "warn": False,
        "CET": 0.0,
    }


def calibration_parameters() -> dict:
    """Return one deterministic parameter vector accepted by ``spot_setup``."""
    return {
        "lr_temp": -0.006,
        "lr_prec": 0.0,
        "BETA": 1.0,
        "CET": 0.0,
        "FC": 250.0,
        "K0": 0.055,
        "K1": 0.055,
        "K2": 0.04,
        "LP": 0.7,
        "MAXBAS": 3.0,
        "PERC": 1.5,
        "UZL": 120.0,
        "PCORR": 1.0,
        "TT_snow": 0.0,
        "TT_diff": 2.0,
        "CFMAX_snow": 2.5,
        "CFMAX_rel": 2.0,
        "SFCF": 0.7,
        "CWH": 0.1,
        "AG": 0.7,
        "CFR": 0.15,
    }


def make_synthetic_observations() -> pd.DataFrame:
    """Return non-constant runoff data used only to exercise mspot scoring."""
    dates = pd.date_range(SIMULATION_START, SIMULATION_END, freq="D")
    day_of_year = dates.dayofyear.to_numpy()
    runoff_mm = 0.8 + 0.3 * np.sin(
        2 * np.pi * (day_of_year - 60) / 365.25
    )
    runoff_m3_per_second = (
        runoff_mm * AREA_CATCHMENT * 1_000_000 / 86_400 / 1000
    )
    return pd.DataFrame({"Date": dates, "Qobs": runoff_m3_per_second})


def load_synthetic_reference(name: str):
    """Load one maintained synthetic public-API output."""
    manifest = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    reference_path = REFERENCE_DIRECTORY / manifest["references"][name]["file"]
    with reference_path.open("rb") as handle:
        return pickle.load(handle)
