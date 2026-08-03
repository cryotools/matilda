"""Loading and execution helpers for the maintained reference scenario."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from matilda.core import matilda_simulation


INPUT_DIRECTORY = Path(__file__).parent / "test_input"
REFERENCE_PATH = INPUT_DIRECTORY / "baseline_output.pickle"
MANIFEST_PATH = INPUT_DIRECTORY / "baseline_manifest.json"


def load_reference():
    with REFERENCE_PATH.open("rb") as handle:
        return pickle.load(handle)


def run_reference_scenario():
    """Run the standard 2000-2020 glacier-evolution scenario from fresh inputs."""
    with (INPUT_DIRECTORY / "parameters.yml").open(encoding="utf-8") as handle:
        parameters = yaml.safe_load(handle)
    with (INPUT_DIRECTORY / "settings.yml").open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)

    forcing = pd.read_csv(INPUT_DIRECTORY / "era5.csv")
    observations = pd.read_csv(INPUT_DIRECTORY / "obs_runoff_example.csv")
    glacier_profile = pd.read_csv(INPUT_DIRECTORY / "glacier_profile.csv")

    captured_output = io.StringIO()
    try:
        with redirect_stdout(captured_output):
            return matilda_simulation(
                input_df=forcing,
                obs=observations,
                **settings,
                **parameters,
                glacier_profile=glacier_profile,
            )
    finally:
        plt.close("all")
