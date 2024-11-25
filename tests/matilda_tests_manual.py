# -*- coding: UTF-8 -*-
"""
Script: MATILDA Model Testing and Comparison Routine
Author: Phillip Schuster
Date: 2024-11-19

Description:
-------------
This script implements a manual testing routine for the MATILDA glacio-hydrological model to ensure consistency and correctness
of the model output. It compares the results of the current model run with a saved baseline and identifies differences
in key outputs such as time series data, performance metrics, and summary plots.

Key Features:
-------------
1. **Baseline Management**:
   - Load a pre-saved baseline output from a pickle file for comparison.
   - If no baseline exists, creates a new one and saves it for future runs.

2. **Output Comparison**:
   - Compares key elements of the model output:
     a. Time series data (DataFrame) with column-wise summaries of mean, sum, and maximum differences.
     b. KGE (Kling-Gupta Efficiency) score for model performance.

3. **Visual Comparison**:
   - Displays a side-by-side comparison of the summary plots from the current model run and the baseline.

4. **Interactive Environment Support**:
   - Automatically detects the working environment (script execution or interactive mode) and adjusts the working directory accordingly.

Dependencies:
-------------
- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `yaml`, `pickle`
- MATILDA model modules (`matilda.core`, `matilda.mspot_glacier`)

Usage:
------
1. Ensure the required input files are in the specified directory (`parameters.yml`, `settings.yml`, `era5.csv`, `obs_runoff_example.csv`, `swe.csv`, `glacier_profile.csv`).
2. Run the script as a standalone Python file.
3. The script will:
   - Compare the current model output with the baseline.
   - Print a summary of differences in the terminal.
   - Display visual comparisons of summary plots.

Notes:
------
- The baseline file (`baseline_output.pickle`) is essential for comparison. If not found, a new one will be created.
- Tolerance for numerical comparisons is set to 1e-3 but can be adjusted as needed.

"""

import matplotlib.pyplot as plt
import yaml
import os
import numpy as np
import pandas as pd
import pickle

from matilda.core import matilda_simulation
from matilda.mspot_glacier import HiddenPrints

try:
    # Case 1: Script execution
    home = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Case 2: Interactive environment (e.g., Jupyter, IDE)
    cwd = os.getcwd()
    if os.path.basename(cwd) == "tests":
        home = cwd
    elif os.path.basename(cwd) == "matilda":
        home = os.path.join(cwd, "tests")
    else:
        raise ValueError(
            "Unknown working directory. Please specify the directory manually."
        )

print(f"The home directory is set to: {home}")


def load_baseline(file_path):
    """Load the baseline output from a pickle file."""
    with open(file_path, "rb") as f:
        baseline = pickle.load(f)
        return baseline


def save_baseline(output, file_path):
    """Save the model output as a baseline to a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(output, f)


def compare_dataframes(df1, df2, tolerance=1e-3):
    """
    Compare two dataframes column by column and return a summary of differences.

    Parameters:
        df1 (pd.DataFrame): New DataFrame.
        df2 (pd.DataFrame): Baseline DataFrame.
        tolerance (float): Tolerance for detecting significant differences.

    Returns:
        dict: A summary of differences (mean, sum) for each column.
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


def compare_output(new_output, baseline_output, tolerance=1e-3):
    """
    Compare the outputs of the model.

    Parameters:
        new_output: The output of the new simulation.
        baseline_output: The baseline simulation output.
        tolerance (float): Tolerance for detecting differences.

    Returns:
        dict: Summary of differences.
    """
    differences = {}

    # Compare the first element: DataFrame
    new_df = pd.DataFrame(new_output[0])
    baseline_df = pd.DataFrame(baseline_output[0])
    df_differences = compare_dataframes(new_df, baseline_df, tolerance)
    if df_differences:
        differences["DataFrame"] = df_differences

    # Compare the third element: KGE metric (float64)
    if not np.isclose(new_output[2], baseline_output[2], atol=tolerance):
        differences["KGE"] = {
            "new_value": round(new_output[2], 3),
            "baseline_value": round(baseline_output[2], 3),
            "absolute_diff": round(abs(new_output[2] - baseline_output[2]), 3),
        }

    return differences


def plot_comparison(new_plot, baseline_plot):
    """Display the plots for visual comparison with larger title font sizes."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Render the new plot in the first subplot
    axs[0].set_title("New Plot", fontsize=16)  # Increased font size
    new_plot_canvas = new_plot.canvas
    new_plot_canvas.draw()
    new_plot_image = np.frombuffer(new_plot_canvas.tostring_rgb(), dtype="uint8")
    new_plot_image = new_plot_image.reshape(
        new_plot_canvas.get_width_height()[::-1] + (3,)
    )
    axs[0].imshow(new_plot_image)
    axs[0].axis("off")

    # Render the baseline plot in the second subplot
    axs[1].set_title("Baseline Plot", fontsize=16)  # Increased font size
    baseline_plot_canvas = baseline_plot.canvas
    baseline_plot_canvas.draw()
    baseline_plot_image = np.frombuffer(
        baseline_plot_canvas.tostring_rgb(), dtype="uint8"
    )
    baseline_plot_image = baseline_plot_image.reshape(
        baseline_plot_canvas.get_width_height()[::-1] + (3,)
    )
    axs[1].imshow(baseline_plot_image)
    axs[1].axis("off")

    plt.tight_layout()


def main():
    # Load inputs for testing
    with open(f"{home}/test_input/parameters.yml", "r") as f:
        parameters = yaml.safe_load(f)

    with open(f"{home}/test_input/settings.yml", "r") as f:
        settings = yaml.safe_load(f)

    data = pd.read_csv(f"{home}/test_input/era5.csv")
    obs = pd.read_csv(f"{home}/test_input/obs_runoff_example.csv")
    swe = pd.read_csv(f"{home}/test_input/swe.csv")
    glacier_profile = pd.read_csv(f"{home}/test_input/glacier_profile.csv")

    # Point to baseline results if available
    baseline_file = f"{home}/test_input/baseline_output.pickle"

    with HiddenPrints():
        new_output = matilda_simulation(
            input_df=data,
            obs=obs,
            **settings,
            **parameters,
            glacier_profile=glacier_profile,
        )

    # Load baseline
    try:
        baseline_output = load_baseline(baseline_file)
    except FileNotFoundError:
        print("Baseline not found. Creating a new one.")
        save_baseline(new_output, baseline_file)
        return

    # Compare outputs
    differences = compare_output(new_output, baseline_output)

    if differences:
        print("Differences detected:")
        for key, value in differences.items():
            if key == "DataFrame":
                print("\nDataFrame Differences (summary):")
                # Convert differences to DataFrame
                df_differences = pd.DataFrame(value).T
                print(df_differences)  # Print in tabular form
            else:
                print(f"\n{key}: {value}")
    else:
        print("No significant differences detected.")

    # Plot visual comparison of the eighth element
    new_plot = new_output[7]
    baseline_plot = baseline_output[7]
    plot_comparison(new_plot, baseline_plot)

    # Get the list of all open figure numbers
    open_figures = plt.get_fignums()

    # Close all model figure, except the comparison (workaround for backend 'TkAgg')
    for fig_num in open_figures:
        if fig_num != 7:  # Skip figures 2 and 5
            try:
                plt.close(plt.figure(fig_num))
            except Exception as e:
                print(f"Error closing figure {fig_num}: {e}")

    plt.show()


if __name__ == "__main__":
    main()

## Run manually

# with open(f"{home}/test_input/parameters.yml", "r") as f:
#     parameters = yaml.safe_load(f)
#
# with open(f"{home}/test_input/settings.yml", "r") as f:
#     settings = yaml.safe_load(f)
#
# data = pd.read_csv(f"{home}/test_input/era5.csv")
# obs = pd.read_csv(f"{home}/test_input/obs_runoff_example.csv")
# swe = pd.read_csv(f"{home}/test_input/swe.csv")
# glacier_profile = pd.read_csv(f"{home}/test_input/glacier_profile.csv")
#
# new_output = matilda_simulation(
#     input_df=data, obs=obs, **settings, **parameters, glacier_profile=glacier_profile
# )
#
# new_output[10].show()
