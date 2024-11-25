# -*- coding: UTF-8 -*-
"""
Script: Automated MATILDA Model Testing Routine
Author: Phillip Schuster
Date: 2024-11-25

Description:
-------------
This script implements an **automated testing routine** for the MATILDA glacio-hydrological model using `pytest`. It ensures
consistency and correctness of the model output by comparing the current simulation results with a pre-saved baseline.

Key Features:
-------------
1. **Pytest Integration**:
   - Fully automated testing framework leveraging `pytest` to validate model outputs.
   - Provides concise error messages and integration with CI/CD pipelines for continuous testing.

2. **Baseline Management**:
   - Loads a pre-saved baseline output from a pickle file for comparison.
   - Ensures the baseline remains unchanged during tests.

3. **Output Comparison**:
   - Compares key elements of the model output:
     a. Time series data (DataFrame) with column-wise summaries of mean, sum, and maximum differences.
     b. KGE (Kling-Gupta Efficiency) score for model performance.

4. **Fixtures for Flexibility**:
   - Modular fixtures for dynamically determining file paths, loading input data, and managing baselines.

5. **Improved Diagnostics**:
   - Prints detailed differences in DataFrame outputs if a test fails for easier debugging.
   - Allows better visualization of detected differences in terminal output.

6. **Interactive Environment Support**:
   - Detects working environments dynamically (e.g., IDE, script execution) for seamless path resolution.

Dependencies:
-------------
- Python 3.x
- `pytest`, `numpy`, `pandas`, `matplotlib`, `yaml`, `pickle`
- MATILDA model modules (`matilda.core`, `matilda.mspot_glacier`)

Usage:
------
1. Ensure the required input files are in the specified `test_input` directory:
   - `parameters.yml`
   - `settings.yml`
   - `era5.csv`
   - `obs_runoff_example.csv`
   - `swe.csv`
   - `glacier_profile.csv`
   - `baseline_output.pickle` (manually created baseline)

2. Run the test:
    pytest test_matilda.py


3. The test will:
- Compare the current model output with the baseline.
- Print detailed differences if a test fails.
- Pass if no significant differences are detected.

4. To add this test to a CI/CD pipeline, ensure all required files are available.

Notes:
------
- The baseline file (`baseline_output.pickle`) must be manually created before running the test.
- If the baseline is missing, the test will fail with an appropriate message.
- Tolerance for numerical comparisons is set to `1e-3` but can be adjusted as needed.

"""


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spotpy")
import pytest
import os
import pandas as pd
import pickle
import yaml
import numpy as np
from matilda.core import matilda_simulation
from matilda.mspot_glacier import HiddenPrints


@pytest.fixture
def test_directory():
    """
    Determine the test directory path dynamically.

    Returns:
    --------
    str:
        The absolute path to the test directory.

    Raises:
    -------
    ValueError:
        If the working directory cannot be determined.
    """
    try:
        # Case 1: Script execution
        test_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Case 2: Interactive environment (e.g., Jupyter, IDE)
        cwd = os.getcwd()
        if os.path.basename(cwd) == "tests":
            test_dir = cwd
        elif os.path.basename(cwd) == "matilda":
            test_dir = os.path.join(cwd, "tests")
        else:
            raise ValueError(
                "Unknown working directory. Please specify the directory manually."
            )
    return test_dir


@pytest.fixture
def load_inputs(test_directory):
    """Load test inputs from the specified directory."""
    inputs = {}
    with open(f"{test_directory}/test_input/parameters.yml", "r") as f:
        inputs["parameters"] = yaml.safe_load(f)
    with open(f"{test_directory}/test_input/settings.yml", "r") as f:
        inputs["settings"] = yaml.safe_load(f)
    inputs["data"] = pd.read_csv(f"{test_directory}/test_input/era5.csv")
    inputs["obs"] = pd.read_csv(f"{test_directory}/test_input/obs_runoff_example.csv")
    inputs["swe"] = pd.read_csv(f"{test_directory}/test_input/swe.csv")
    inputs["glacier_profile"] = pd.read_csv(
        f"{test_directory}/test_input/glacier_profile.csv"
    )
    return inputs


@pytest.fixture
def baseline_file(test_directory):
    """Fixture for the baseline file path."""
    return f"{test_directory}/test_input/baseline_output.pickle"


def load_baseline(file_path):
    """Load the baseline output from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_baseline(output, file_path):
    """Save the model output as a baseline to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(output, f)


def compare_dataframes(df1, df2, tolerance=1e-3):
    """
    Compare two dataframes column by column and return a summary of differences.
    """
    summary = {}
    for col in df1.columns:
        if col in df2.columns:
            diff = np.abs(df1[col] - df2[col])
            if (diff > tolerance).any():
                summary[col] = {
                    "mean_diff": round(diff.mean(), 3),
                    "sum_diff": round(diff.sum(), 3),
                    "max_diff": round(diff.max(), 3),
                }
    return summary


def test_simulation_baseline(load_inputs, baseline_file):
    """
    Test the MATILDA simulation against a pre-saved baseline.
    """
    inputs = load_inputs

    with HiddenPrints():
        new_output = matilda_simulation(
            input_df=inputs["data"],
            obs=inputs["obs"],
            **inputs["settings"],
            **inputs["parameters"],
            glacier_profile=inputs["glacier_profile"],
        )

    try:
        baseline_output = load_baseline(baseline_file)
    except FileNotFoundError:
        pytest.fail(
            "Baseline not found. Ensure you have created a baseline manually before running this test."
        )

    # Compare DataFrame outputs
    new_df = pd.DataFrame(new_output[0])
    baseline_df = pd.DataFrame(baseline_output[0])
    df_differences = compare_dataframes(new_df, baseline_df)

    if df_differences:
        # Convert differences to a DataFrame for better visualization
        diff_df = pd.DataFrame(df_differences).T
        print("\nDataFrame Differences (detailed):")
        print(diff_df.to_string())  # Use to_string() for a readable format

    # Assert no differences
    assert (
        not df_differences
    ), f"Significant differences to baseline detected: {df_differences}"

    # Compare KGE metric
    assert np.isclose(
        new_output[2], baseline_output[2], atol=1e-3
    ), f"KGE mismatch: New {new_output[2]} vs Baseline {baseline_output[2]}"
