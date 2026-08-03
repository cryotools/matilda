"""Detailed numerical impact reporting for MATILDA model output comparisons."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal


FRAME_OUTPUTS = {
    0: "compact_daily",
    1: "full_daily",
    3: "summary_statistics",
    4: "glacier_elevation_distribution",
    5: "annual_glacier_evolution",
}


def classify_process(output_name: str, variable: object) -> str:
    """Assign a broad process label for sorting impact reports."""
    name = str(variable).lower()
    if output_name.startswith("glacier_") or any(
        token in name
        for token in (
            "ddm_",
            "smb",
            "on_glaciers",
            "glacier_area",
            "glacier_elev",
            "glacier_mass",
            "glacier_vol",
            "ice_melt",
        )
    ):
        return "glacier"
    if any(token in name for token in ("runoff", "q_hbv", "q_total", "qobs")):
        return "runoff"
    if any(token in name for token in ("groundwater", "upper_gw", "lower_gw")):
        return "groundwater"
    if "soil" in name:
        return "soil"
    if any(token in name for token in ("evap", "aet", "hbv_pe")):
        return "evapotranspiration"
    if any(token in name for token in ("snow", "melt", "refreez")):
        return "snow_and_melt"
    if any(token in name for token in ("prec", "rain")):
        return "precipitation"
    if any(token in name for token in ("temp", "pdd")):
        return "temperature"
    return "diagnostic"


def _safe_float(value) -> float:
    return float(value) if pd.notna(value) else np.nan


def _sum(series: pd.Series) -> float:
    return _safe_float(series.sum(min_count=1))


def _comparison_mask(
    reference: pd.Series,
    current: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    reference_aligned, current_aligned = reference.align(current, join="outer")
    same = reference_aligned.eq(current_aligned) | (
        reference_aligned.isna() & current_aligned.isna()
    )
    reference_present = pd.Series(
        reference_aligned.index.isin(reference.index), index=reference_aligned.index
    )
    current_present = pd.Series(
        current_aligned.index.isin(current.index), index=current_aligned.index
    )
    same &= reference_present & current_present
    return reference_aligned, current_aligned, same


def _missing_column_row(
    output_name: str,
    variable: object,
    status: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    return {
        "output": output_name,
        "process": classify_process(output_name, variable),
        "variable": str(variable),
        "status": status,
        "reference_dtype": (
            str(reference[variable].dtype) if variable in reference else ""
        ),
        "current_dtype": str(current[variable].dtype) if variable in current else "",
        "index_equal": reference.index.equals(current.index),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_missing": (
            int(reference[variable].isna().sum()) if variable in reference else np.nan
        ),
        "current_missing": (
            int(current[variable].isna().sum()) if variable in current else np.nan
        ),
        "changed_count": max(len(reference), len(current)),
        "max_abs_change": np.nan,
        "mean_abs_change": np.nan,
        "rmse": np.nan,
        "reference_sum": np.nan,
        "current_sum": np.nan,
        "sum_change": np.nan,
        "relative_sum_change_percent": np.nan,
        "reference_mean": np.nan,
        "current_mean": np.nan,
        "mean_change": np.nan,
        "first_changed_index": "",
    }


def _column_impact(
    output_name: str,
    variable: object,
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> dict:
    reference_series, current_series, same = _comparison_mask(
        reference[variable], current[variable]
    )
    dtype_equal = reference[variable].dtype == current[variable].dtype
    index_equal = reference.index.equals(current.index)
    changed_count = int((~same).sum())
    first_changed = ""
    if changed_count:
        first_changed = str(same.index[~same][0])

    numeric = is_numeric_dtype(reference_series) and is_numeric_dtype(current_series)
    if numeric:
        delta = current_series - reference_series
        absolute = delta.abs()
        reference_sum = _sum(reference_series)
        current_sum = _sum(current_series)
        sum_change = current_sum - reference_sum
        relative_sum_change = (
            sum_change / abs(reference_sum) * 100
            if reference_sum != 0 and np.isfinite(reference_sum)
            else np.nan
        )
        max_abs_change = _safe_float(absolute.max())
        mean_abs_change = _safe_float(absolute.mean())
        rmse = _safe_float(np.sqrt((delta**2).mean()))
        reference_mean = _safe_float(reference_series.mean())
        current_mean = _safe_float(current_series.mean())
        mean_change = current_mean - reference_mean
    else:
        max_abs_change = mean_abs_change = rmse = np.nan
        reference_sum = current_sum = sum_change = np.nan
        relative_sum_change = np.nan
        reference_mean = current_mean = mean_change = np.nan

    status = (
        "unchanged"
        if changed_count == 0 and dtype_equal and index_equal
        else "changed"
    )
    return {
        "output": output_name,
        "process": classify_process(output_name, variable),
        "variable": str(variable),
        "status": status,
        "reference_dtype": str(reference[variable].dtype),
        "current_dtype": str(current[variable].dtype),
        "index_equal": index_equal,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "reference_missing": int(reference[variable].isna().sum()),
        "current_missing": int(current[variable].isna().sum()),
        "changed_count": changed_count,
        "max_abs_change": max_abs_change,
        "mean_abs_change": mean_abs_change,
        "rmse": rmse,
        "reference_sum": reference_sum,
        "current_sum": current_sum,
        "sum_change": sum_change,
        "relative_sum_change_percent": relative_sum_change,
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "mean_change": mean_change,
        "first_changed_index": first_changed,
    }


def build_variable_impact_report(current_output, reference_output) -> pd.DataFrame:
    """Return one quantitative comparison row for every maintained variable."""
    rows = []
    for position, output_name in FRAME_OUTPUTS.items():
        reference = reference_output[position]
        current = current_output[position]
        variables = list(reference.columns) + [
            variable for variable in current.columns if variable not in reference.columns
        ]
        for variable in variables:
            if variable not in current.columns:
                rows.append(
                    _missing_column_row(
                        output_name, variable, "missing", reference, current
                    )
                )
            elif variable not in reference.columns:
                rows.append(
                    _missing_column_row(output_name, variable, "added", reference, current)
                )
            else:
                rows.append(_column_impact(output_name, variable, reference, current))

    reference_kge = float(reference_output[2])
    current_kge = float(current_output[2])
    delta = current_kge - reference_kge
    rows.append(
        {
            "output": "model_efficiency",
            "process": "runoff",
            "variable": "KGE",
            "status": "unchanged" if current_kge == reference_kge else "changed",
            "reference_dtype": type(reference_output[2]).__name__,
            "current_dtype": type(current_output[2]).__name__,
            "index_equal": True,
            "reference_rows": 1,
            "current_rows": 1,
            "reference_missing": int(np.isnan(reference_kge)),
            "current_missing": int(np.isnan(current_kge)),
            "changed_count": int(current_kge != reference_kge),
            "max_abs_change": abs(delta),
            "mean_abs_change": abs(delta),
            "rmse": abs(delta),
            "reference_sum": reference_kge,
            "current_sum": current_kge,
            "sum_change": delta,
            "relative_sum_change_percent": (
                delta / abs(reference_kge) * 100 if reference_kge != 0 else np.nan
            ),
            "reference_mean": reference_kge,
            "current_mean": current_kge,
            "mean_change": delta,
            "first_changed_index": "metric" if delta else "",
        }
    )
    return pd.DataFrame(rows)


def build_annual_impact_report(current_output, reference_output) -> pd.DataFrame:
    """Summarize pointwise, mean, and total changes for each calendar year."""
    rows = []
    for position, output_name in FRAME_OUTPUTS.items():
        reference = reference_output[position]
        current = current_output[position]
        if not isinstance(reference.index, pd.DatetimeIndex) or not isinstance(
            current.index, pd.DatetimeIndex
        ):
            continue
        for variable in reference.columns.intersection(current.columns, sort=False):
            if not (
                is_numeric_dtype(reference[variable])
                and is_numeric_dtype(current[variable])
            ):
                continue
            reference_series, current_series, same = _comparison_mask(
                reference[variable], current[variable]
            )
            data = pd.DataFrame(
                {
                    "reference": reference_series,
                    "current": current_series,
                    "same": same,
                }
            )
            for year, group in data.groupby(data.index.year):
                delta = group["current"] - group["reference"]
                reference_sum = _sum(group["reference"])
                current_sum = _sum(group["current"])
                reference_mean = _safe_float(group["reference"].mean())
                current_mean = _safe_float(group["current"].mean())
                rows.append(
                    {
                        "output": output_name,
                        "process": classify_process(output_name, variable),
                        "variable": str(variable),
                        "year": int(year),
                        "changed_count": int((~group["same"]).sum()),
                        "max_abs_point_change": _safe_float(delta.abs().max()),
                        "rmse": _safe_float(np.sqrt((delta**2).mean())),
                        "reference_sum": reference_sum,
                        "current_sum": current_sum,
                        "sum_change": current_sum - reference_sum,
                        "reference_mean": reference_mean,
                        "current_mean": current_mean,
                        "mean_change": current_mean - reference_mean,
                    }
                )
    return pd.DataFrame(rows)


def exact_output_errors(current_output, reference_output) -> list[str]:
    """Return concise structural or numerical errors for maintained outputs."""
    errors = []
    if len(current_output) != len(reference_output):
        errors.append(
            f"public output length: current={len(current_output)}, "
            f"reference={len(reference_output)}"
        )
    for position, output_name in FRAME_OUTPUTS.items():
        try:
            assert_frame_equal(
                current_output[position],
                reference_output[position],
                check_exact=True,
                check_dtype=True,
                check_index_type=True,
                check_column_type=True,
                check_names=True,
                check_freq=True,
            )
        except AssertionError as error:
            message = "\n".join(str(error).splitlines()[:8])
            errors.append(f"{output_name}: {message}")
    if not np.array_equal(
        np.asarray(current_output[2]),
        np.asarray(reference_output[2]),
        equal_nan=True,
    ):
        errors.append(
            f"KGE: current={current_output[2]!r}, reference={reference_output[2]!r}"
        )
    return errors


def format_impact_summary(report: pd.DataFrame, maximum_rows: int = 40) -> str:
    """Format the largest changed variables for a pytest failure message."""
    changed = report[report["status"] != "unchanged"].copy()
    if changed.empty:
        return "No per-variable value changes were found."
    changed["sort_change"] = changed["max_abs_change"].fillna(-1)
    changed = changed.sort_values(
        ["sort_change", "changed_count"], ascending=False
    ).head(maximum_rows)
    return changed[
        [
            "output",
            "process",
            "variable",
            "status",
            "changed_count",
            "max_abs_change",
            "rmse",
            "sum_change",
            "first_changed_index",
        ]
    ].to_string(index=False)


def write_impact_reports(
    directory: Path,
    variable_report: pd.DataFrame,
    annual_report: pd.DataFrame,
) -> None:
    """Write machine-readable diagnostics for scientific impact assessment."""
    directory.mkdir(parents=True, exist_ok=True)
    variable_report.to_csv(
        directory / "variable_impact.csv", index=False, float_format="%.17g"
    )
    annual_report.to_csv(
        directory / "annual_impact.csv", index=False, float_format="%.17g"
    )
    changed = variable_report[variable_report["status"] != "unchanged"]
    summary = {
        "python": platform.python_version(),
        "matilda": metadata.version("matilda"),
        "compared_variables": int(len(variable_report)),
        "changed_variables": int(len(changed)),
        "changed_processes": sorted(changed["process"].unique().tolist()),
        "reports": ["variable_impact.csv", "annual_impact.csv"],
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
