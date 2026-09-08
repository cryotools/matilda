"""Synthetic public-API tests for distinct MATILDA model modes."""

from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from matilda.core import matilda_simulation
from tests.impact import (
    build_annual_impact_report,
    build_variable_impact_report,
    exact_output_errors,
    format_impact_summary,
    write_impact_reports,
)
from tests.synthetic import (
    REFERENCE_DIRECTORY,
    REFERENCE_MANIFEST_PATH,
    SIMULATION_END,
    SIMULATION_START,
    SYNTHETIC_FRAME_OUTPUTS,
    load_synthetic_reference,
    make_synthetic_forcing,
    make_synthetic_glacier_profile,
    model_settings,
)


pytestmark = pytest.mark.model_modes


def _run_synthetic_model(area_glac: float):
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    try:
        with redirect_stdout(io.StringIO()):
            output = matilda_simulation(
                forcing,
                **model_settings(area_glac=area_glac),
            )
    finally:
        plt.close("all")
    return output, forcing, forcing_before


@pytest.fixture(scope="module")
def zero_glacier_run():
    return _run_synthetic_model(area_glac=0.0)


@pytest.fixture(scope="module")
def fixed_glacier_run():
    return _run_synthetic_model(area_glac=20.0)


def test_synthetic_reference_integrity():
    manifest = json.loads(REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["core_baseline_commit"] == (
        "7a7625c9365e21dfe1b9e267b08139774e37c446"
    )
    assert manifest["model_version"] == "1.0.2"
    for reference in manifest["references"].values():
        reference_path = REFERENCE_DIRECTORY / reference["file"]
        assert sha256(reference_path.read_bytes()).hexdigest() == reference["sha256"]


def _assert_synthetic_reference(
    name,
    current_output,
    impact_report_directory,
):
    reference_output = load_synthetic_reference(name)
    variable_report = build_variable_impact_report(
        current_output,
        reference_output,
        frame_outputs=SYNTHETIC_FRAME_OUTPUTS,
        include_model_efficiency=False,
    )
    annual_report = build_annual_impact_report(
        current_output,
        reference_output,
        frame_outputs=SYNTHETIC_FRAME_OUTPUTS,
    )

    if impact_report_directory is not None:
        write_impact_reports(
            impact_report_directory / name,
            variable_report,
            annual_report,
        )

    errors = exact_output_errors(
        current_output,
        reference_output,
        frame_outputs=SYNTHETIC_FRAME_OUTPUTS,
        include_model_efficiency=False,
    )
    non_frame_outputs = (
        (2, "model efficiency"),
        (4, "lookup table"),
        (5, "glacier changes"),
    )
    for position, label in non_frame_outputs:
        if current_output[position] != reference_output[position]:
            errors.append(
                f"{label}: current={current_output[position]!r}, "
                f"reference={reference_output[position]!r}"
            )
    changed = variable_report[variable_report["status"] != "unchanged"]
    if errors or not changed.empty:
        details = "\n".join(errors)
        summary = format_impact_summary(variable_report)
        pytest.fail(
            f"Synthetic {name} output differs from its maintained reference.\n"
            f"{details}\n\nChanged variables:\n{summary}"
        )


def test_synthetic_modes_match_exact_references(
    zero_glacier_run,
    fixed_glacier_run,
    impact_report_directory,
):
    _assert_synthetic_reference(
        "zero_glacier",
        zero_glacier_run[0],
        impact_report_directory,
    )
    _assert_synthetic_reference(
        "fixed_glacier",
        fixed_glacier_run[0],
        impact_report_directory,
    )


def test_zero_glacier_mode_is_complete_and_does_not_mutate_input(zero_glacier_run):
    output, forcing, forcing_before = zero_glacier_run
    compact, full = output[:2]
    expected_index = pd.date_range(SIMULATION_START, SIMULATION_END, freq="D")

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert full.index.equals(expected_index)
    assert len(full) == 731
    assert pd.Timestamp("2000-02-29") in full.index
    assert not any(column.startswith("DDM_") for column in full.columns)
    assert full["Q_Total"].equals(full["Q_HBV"])
    assert np.isfinite(full.select_dtypes(include=np.number)).all().all()

    nonnegative = [
        "HBV_prec",
        "HBV_rain",
        "HBV_snow",
        "HBV_pe",
        "HBV_snowpack",
        "HBV_soil_moisture",
        "HBV_AET",
        "HBV_refreezing",
        "HBV_upper_gw",
        "HBV_lower_gw",
        "HBV_melt_off_glacier",
        "Q_HBV",
        "Q_Total",
    ]
    assert (full[nonnegative] >= 0).all().all()
    np.testing.assert_allclose(
        full["HBV_prec"],
        full["HBV_rain"] + full["HBV_snow"],
        atol=0.002,
    )
    assert (compact["runoff"] >= 0).all()
    assert output[4] == "No lookup table generated"
    assert output[5] == "No glacier changes calculated"


def test_fixed_glacier_mode_preserves_component_balances(fixed_glacier_run):
    output, forcing, forcing_before = fixed_glacier_run
    compact, full = output[:2]
    glacier_fraction = 0.2

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert np.isfinite(full.select_dtypes(include=np.number)).all().all()
    assert (compact["total_runoff"] >= 0).all()
    assert (compact["runoff_from_glaciers"] >= 0).all()
    assert (compact["runoff_without_glaciers"] >= 0).all()

    np.testing.assert_allclose(
        full["Q_Total"],
        full["Q_HBV"] + full["Q_DDM_scaled"],
        atol=0.002,
    )
    np.testing.assert_allclose(
        full["Prec_total"],
        full["DDM_rain_scaled"]
        + full["DDM_snow_scaled"]
        + full["HBV_rain"]
        + full["HBV_snow"],
        atol=0.003,
    )
    np.testing.assert_allclose(
        full["Melt_total"],
        full["DDM_total_melt_scaled"] + full["HBV_melt_off_glacier"],
        atol=0.002,
    )

    for variable in (
        "DDM_prec",
        "DDM_rain",
        "DDM_snow",
        "DDM_accumulation_rate",
        "DDM_ice_melt",
        "DDM_snow_melt",
        "DDM_total_melt",
        "DDM_refreezing",
        "DDM_glacier_reservoir",
        "Q_DDM",
    ):
        np.testing.assert_allclose(
            full[f"{variable}_scaled"],
            full[variable] * glacier_fraction,
            atol=0.001,
        )

    assert output[4] == "No lookup table generated"
    assert output[5] == "No glacier changes calculated"


def test_fixed_glacier_mode_is_exactly_repeatable(fixed_glacier_run):
    first_output = fixed_glacier_run[0]
    second_output = _run_synthetic_model(area_glac=20.0)[0]

    for position in (0, 1, 3):
        assert_frame_equal(
            first_output[position],
            second_output[position],
            check_exact=True,
        )
    assert first_output[2] == second_output[2]
    assert first_output[4:] == second_output[4:]


def test_zero_glacier_evolution_matches_standard_zero_glacier_mode(
    zero_glacier_run,
):
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    profile = make_synthetic_glacier_profile()
    profile_before = profile.copy(deep=True)
    settings = model_settings(area_glac=0.0)
    settings["elev_rescaling"] = True

    try:
        with redirect_stdout(io.StringIO()):
            evolving_output = matilda_simulation(
                forcing,
                glacier_profile=profile,
                **settings,
            )
    finally:
        plt.close("all")

    standard_output = zero_glacier_run[0]
    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert_frame_equal(profile, profile_before, check_exact=True)
    for position in (0, 1, 3):
        assert_frame_equal(
            evolving_output[position],
            standard_output[position],
            check_exact=True,
        )
    assert evolving_output[2] == standard_output[2]
    assert evolving_output[4:] == standard_output[4:]


def test_one_water_year_glacier_evolution_completes():
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    profile = make_synthetic_glacier_profile()
    profile_before = profile.copy(deep=True)
    settings = model_settings(area_glac=20.0)
    settings.update(
        {
            "sim_end": "2000-09-30",
            "elev_rescaling": True,
        }
    )

    try:
        with redirect_stdout(io.StringIO()):
            output = matilda_simulation(
                forcing,
                glacier_profile=profile,
                **settings,
            )
    finally:
        plt.close("all")

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert_frame_equal(
        profile.loc[:, profile_before.columns],
        profile_before,
        check_exact=True,
    )
    assert len(output[1]) == 274
    assert np.isfinite(output[1].select_dtypes(include=np.number)).all().all()
    assert output[5]["glacier_area"].tolist() == [20.0]


def test_glacier_evolution_handles_loss_in_first_update():
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    profile = make_synthetic_glacier_profile()
    profile["WE"] = 1.0
    profile_before = profile.copy(deep=True)
    settings = model_settings(area_glac=20.0)
    settings["elev_rescaling"] = True

    try:
        with redirect_stdout(io.StringIO()):
            output = matilda_simulation(
                forcing,
                glacier_profile=profile,
                **settings,
            )
    finally:
        plt.close("all")

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert_frame_equal(
        profile.loc[:, profile_before.columns],
        profile_before,
        check_exact=True,
    )
    assert output[5]["glacier_area"].iloc[1] == 0
    assert np.isfinite(output[5]["glacier_elev"]).all()
    assert np.isfinite(output[1].select_dtypes(include=np.number)).all().all()
