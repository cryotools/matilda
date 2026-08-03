"""Unit tests for numerical impact diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tests.impact import (
    build_annual_impact_report,
    build_variable_impact_report,
    exact_output_errors,
    write_impact_reports,
)


def _synthetic_outputs():
    index = pd.to_datetime(["2000-12-31", "2001-12-31"])
    reference_changed = pd.DataFrame(
        {
            "flux": [1.0, 2.0],
            "state": [np.nan, 4.0],
            "removed": [3.0, 3.0],
        },
        index=index,
    )
    current_changed = pd.DataFrame(
        {
            "flux": [1.5, 1.0],
            "state": [np.nan, 5.0],
            "added": [7.0, 8.0],
        },
        index=index,
    )
    stable = pd.DataFrame({"stable": [1.0, 2.0]}, index=index)
    stable_distribution = stable.reset_index(drop=True)
    reference = [
        reference_changed,
        stable.copy(),
        np.float64(0.5),
        stable.copy(),
        stable_distribution.copy(),
        stable.copy(),
    ]
    current = [
        current_changed,
        stable.copy(),
        np.float64(0.4),
        stable.copy(),
        stable_distribution.copy(),
        stable.copy(),
    ]
    return current, reference


def test_impact_reports_quantify_controlled_changes(tmp_path):
    current, reference = _synthetic_outputs()

    variable_report = build_variable_impact_report(current, reference)
    annual_report = build_annual_impact_report(current, reference)
    flux = variable_report.query(
        "output == 'compact_daily' and variable == 'flux'"
    ).iloc[0]

    assert flux["status"] == "changed"
    assert flux["changed_count"] == 2
    assert flux["max_abs_change"] == pytest.approx(1.0)
    assert flux["mean_abs_change"] == pytest.approx(0.75)
    assert flux["rmse"] == pytest.approx(np.sqrt(0.625))
    assert flux["sum_change"] == pytest.approx(-0.5)
    assert flux["first_changed_index"] == "2000-12-31 00:00:00"

    assert variable_report.query("variable == 'removed'").iloc[0]["status"] == (
        "missing"
    )
    assert variable_report.query("variable == 'added'").iloc[0]["status"] == (
        "added"
    )
    assert variable_report.query("variable == 'KGE'").iloc[0]["sum_change"] == (
        pytest.approx(-0.1)
    )

    annual_flux = annual_report.query(
        "output == 'compact_daily' and variable == 'flux'"
    ).set_index("year")
    assert annual_flux.loc[2000, "sum_change"] == pytest.approx(0.5)
    assert annual_flux.loc[2001, "sum_change"] == pytest.approx(-1.0)

    errors = exact_output_errors(current, reference)
    assert any(error.startswith("compact_daily:") for error in errors)
    assert any(error.startswith("KGE:") for error in errors)

    write_impact_reports(tmp_path, variable_report, annual_report)
    assert (tmp_path / "variable_impact.csv").is_file()
    assert (tmp_path / "annual_impact.csv").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["changed_variables"] == 5
