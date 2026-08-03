"""Model output consistency tests for the maintained reference scenario."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import platform

from matplotlib.figure import Figure as MatplotlibFigure
import pytest
from plotly.graph_objects import Figure as PlotlyFigure

from tests.impact import (
    FRAME_OUTPUTS,
    build_annual_impact_report,
    build_variable_impact_report,
    exact_output_errors,
    format_impact_summary,
    write_impact_reports,
)
from tests.scenario import REFERENCE_PATH


def test_reference_integrity(reference_manifest):
    """Prevent the numerical reference from being replaced accidentally."""
    digest = sha256(REFERENCE_PATH.read_bytes()).hexdigest()
    assert digest == reference_manifest["reference_sha256"]
    assert reference_manifest["core_baseline_commit"] == (
        "7a7625c9365e21dfe1b9e267b08139774e37c446"
    )


def test_reference_environment(reference_manifest):
    """Run consistency checks with the recorded scientific dependencies."""
    assert platform.python_version() == reference_manifest["python"]
    assert metadata.version("matilda") == reference_manifest["model_version"]
    expected_dependencies = {
        **reference_manifest["dependencies"],
        **reference_manifest["transitive_test_dependencies"],
    }
    installed = {
        package: metadata.version(package) for package in expected_dependencies
    }
    assert installed == expected_dependencies


@pytest.mark.model_consistency
def test_public_output_contract(current_output, reference_manifest):
    """Protect the positions and basic types of the public list result."""
    assert isinstance(current_output, list)
    assert len(current_output) == reference_manifest["public_output_length"]

    for position in FRAME_OUTPUTS:
        assert hasattr(current_output[position], "columns")
        assert hasattr(current_output[position], "index")

    assert isinstance(current_output[6], MatplotlibFigure)
    assert isinstance(current_output[7], MatplotlibFigure)
    assert isinstance(current_output[8], MatplotlibFigure)
    assert isinstance(current_output[9], PlotlyFigure)
    assert isinstance(current_output[10], PlotlyFigure)


@pytest.mark.model_consistency
def test_full_model_output_consistency(
    current_output,
    reference_output,
    impact_report_directory,
):
    """Compare every maintained numerical output and produce impact diagnostics."""
    variable_report = build_variable_impact_report(current_output, reference_output)
    annual_report = build_annual_impact_report(current_output, reference_output)

    if impact_report_directory is not None:
        write_impact_reports(
            impact_report_directory,
            variable_report,
            annual_report,
        )

    errors = exact_output_errors(current_output, reference_output)
    changed = variable_report[variable_report["status"] != "unchanged"]
    if errors or not changed.empty:
        details = "\n".join(errors)
        summary = format_impact_summary(variable_report)
        pytest.fail(
            "Full-model output differs from the maintained reference.\n"
            f"{details}\n\nChanged variables:\n{summary}"
        )
