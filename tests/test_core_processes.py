"""Focused equation and bookkeeping tests for core model processes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import xarray as xr

from matilda.core import (
    _load_parameter_data,
    _resample_observations_daily,
    _temperature_adjusted_evaporation,
    calculate_glaciermelt,
    matilda_parameter,
    melt_rates,
    phase_separation,
)
from tests.synthetic import make_synthetic_forcing, model_settings


@pytest.mark.parametrize(
    "timestamps,runoff",
    [
        (
            ["2000-01-01", "2000-01-03", "2000-01-04"],
            [1.25, np.nan, 2.5],
        ),
        (
            ["2000-01-01 06:00", "2000-01-01 18:00", "2000-01-03 00:00"],
            [0.1, np.nan, 2.5],
        ),
    ],
)
def test_daily_observation_aggregation_preserves_missing_values(
    timestamps, runoff
):
    observations = pd.DataFrame(
        {"Qobs": runoff}, index=pd.to_datetime(timestamps)
    )
    expected = observations.resample("D").agg(pd.Series.sum, skipna=False)

    actual = _resample_observations_daily(observations)

    assert_frame_equal(actual, expected, check_exact=True)


def test_daily_observation_aggregation_retains_extra_columns():
    observations = pd.DataFrame(
        {"Qobs": [1.0, 2.0], "quality": ["good", "poor"]},
        index=pd.to_datetime(["2000-01-01", "2000-01-03"]),
    )
    expected = observations.resample("D").agg(pd.Series.sum, skipna=False)

    actual = _resample_observations_daily(observations)

    assert_frame_equal(actual, expected, check_exact=True)


def test_parameter_initialization_derives_dependent_values():
    forcing = make_synthetic_forcing()
    settings = model_settings(area_glac=0.0)
    settings.pop("plots")
    settings.pop("elev_rescaling")

    parameter = matilda_parameter(forcing, **settings)

    assert parameter.area_glac == 0.0
    assert parameter.TT_rain == pytest.approx(
        parameter.TT_snow + parameter.TT_diff
    )
    assert parameter.CFMAX_ice == pytest.approx(
        parameter.CFMAX_snow * parameter.CFMAX_rel
    )


def test_parameter_definitions_are_reused():
    forcing = make_synthetic_forcing()
    settings = model_settings(area_glac=0.0)
    settings.pop("plots")
    settings.pop("elev_rescaling")
    _load_parameter_data.cache_clear()

    try:
        matilda_parameter(forcing, **settings)
        matilda_parameter(forcing, **settings)
        cache_info = _load_parameter_data.cache_info()
    finally:
        _load_parameter_data.cache_clear()

    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_temperature_adjusted_evaporation_matches_daily_calculation():
    forcing = make_synthetic_forcing().set_index("TIMESTAMP")
    temperature = forcing.loc["2000-01-01":"2001-12-31", "T2"]
    evaporation = forcing.loc[temperature.index, "PE"]
    correction_factor = 0.15
    daily_temperature = np.array(
        [
            temperature.loc[temperature.index.dayofyear == day].mean()
            for day in range(1, 367)
        ]
    )
    expected = temperature.index.map(
        lambda timestamp: (
            1
            + correction_factor
            * (
                temperature[timestamp]
                - daily_temperature[timestamp.dayofyear - 1]
            )
        )
        * evaporation[timestamp]
    )

    actual = _temperature_adjusted_evaporation(
        temperature, evaporation, correction_factor
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lat": None}, "No latitude specified"),
        ({"area_cat": None}, "No catchment area specified"),
        ({"area_glac": 101.0}, "exceeds overall catchment area"),
        ({"freq": "H"}, "is not supported"),
    ],
)
def test_parameter_initialization_rejects_invalid_domains(overrides, message):
    forcing = make_synthetic_forcing()
    settings = model_settings(area_glac=0.0)
    settings.pop("plots")
    settings.pop("elev_rescaling")
    settings.update(overrides)

    with pytest.raises(ValueError, match=message):
        matilda_parameter(forcing, **settings)


def test_rain_snow_partition_is_bounded_and_conserves_precipitation():
    parameter = pd.Series({"TT_snow": 0.0, "TT_rain": 2.0})
    forcing = pd.DataFrame(
        {
            "T2": [-1.0, 0.0, 1.0, 2.0, 3.0],
            "RRR": [10.0] * 5,
        }
    )

    rain, snow = phase_separation(forcing, parameter)

    np.testing.assert_allclose(snow, [10.0, 10.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(rain, [0.0, 0.0, 5.0, 10.0, 10.0])
    np.testing.assert_allclose(rain + snow, forcing["RRR"])
    assert rain.between(0, forcing["RRR"]).all()
    assert snow.between(0, forcing["RRR"]).all()


def test_melt_rates_limit_snowmelt_and_preserve_degree_day_energy():
    parameter = pd.Series({"CFMAX_snow": 2.5, "CFMAX_ice": 5.0})
    snow = np.array([0.0, 2.0, 20.0, 1.0])
    pdd = np.array([0.0, 2.0, 2.0, 4.0])

    snow_melt, ice_melt = melt_rates(snow, pdd, parameter)

    potential_snow_melt = parameter.CFMAX_snow * pdd
    np.testing.assert_allclose(snow_melt, [0.0, 2.0, 5.0, 1.0])
    assert np.all(snow_melt <= snow)
    assert np.all(snow_melt >= 0)
    assert np.all(ice_melt >= 0)
    np.testing.assert_allclose(
        snow_melt
        + ice_melt * parameter.CFMAX_snow / parameter.CFMAX_ice,
        potential_snow_melt,
    )


def test_glacier_melt_fluxes_and_storage_are_internally_balanced():
    dates = pd.date_range("2000-01-01", periods=8, freq="D")
    temperature = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0, -2.0, 2.0])
    snow = np.array([4.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    rain = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0])
    dataset = xr.Dataset(
        {
            "temp_mean": ("time", temperature),
            "RRR": ("time", snow + rain),
            "snow": ("time", snow),
            "rain": ("time", rain),
            "pdd": ("time", np.maximum(temperature, 0)),
        },
        coords={"time": dates},
    )
    parameter = pd.Series(
        {
            "CFMAX_snow": 2.5,
            "CFMAX_ice": 5.0,
            "CFR": 0.15,
            "CFR_ice": 0.01,
            "AG": 0.7,
        }
    )

    output = calculate_glaciermelt(dataset, parameter, prints=False)

    np.testing.assert_allclose(
        output["DDM_total_melt"],
        output["DDM_snow_melt"] + output["DDM_ice_melt"],
    )
    np.testing.assert_allclose(
        output["DDM_refreezing"],
        parameter.CFR * output["DDM_snow_melt"]
        + parameter.CFR_ice * output["DDM_ice_melt"],
    )
    liquid_melt = output["DDM_total_melt"] - output["DDM_refreezing"]
    np.testing.assert_allclose(
        output["DDM_smb"], output["DDM_accumulation_rate"] - liquid_melt
    )

    inflow = liquid_melt + output["DDM_rain"]
    reservoir = output["DDM_glacier_reservoir"]
    runoff = output["Q_DDM"]
    signed_variables = ["DDM_temp", "DDM_smb"]
    assert (output.drop(columns=signed_variables) >= 0).all().all()
    assert (runoff <= reservoir).all()
    assert reservoir.iloc[0] == pytest.approx(inflow.iloc[0])
    np.testing.assert_allclose(
        reservoir.iloc[1:].to_numpy(),
        (
            reservoir.iloc[:-1].to_numpy()
            + inflow.iloc[1:].to_numpy()
            - runoff.iloc[:-1].to_numpy()
        ),
    )
