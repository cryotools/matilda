"""Short-run and file-output checks for the public simulation interface."""

from __future__ import annotations

from contextlib import redirect_stdout
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from matilda.core import matilda_simulation
from tests.synthetic import make_synthetic_forcing, model_settings


def test_one_day_simulation_returns_complete_finite_output():
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    settings = model_settings(area_glac=0.0)
    settings.update({"sim_start": "2000-01-01", "sim_end": "2000-01-01"})

    try:
        with redirect_stdout(io.StringIO()):
            output = matilda_simulation(forcing, **settings)
    finally:
        plt.close("all")

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert len(output) == 6
    expected_index = pd.DatetimeIndex(["2000-01-01"], name="TIMESTAMP")
    assert output[1].index.equals(expected_index)
    assert np.isfinite(output[1].select_dtypes(include=np.number)).all().all()


def test_print_output_writes_documented_files(tmp_path):
    forcing = make_synthetic_forcing()
    forcing_before = forcing.copy(deep=True)
    settings = model_settings(area_glac=20.0)
    settings.update(
        {
            "output": str(tmp_path),
            "plots": True,
            "science_plot": False,
            "plot_type": "print",
        }
    )

    try:
        with redirect_stdout(io.StringIO()):
            output = matilda_simulation(forcing, **settings)
    finally:
        plt.close("all")

    assert_frame_equal(forcing, forcing_before, check_exact=True)
    output_directories = list(tmp_path.iterdir())
    assert len(output_directories) == 1
    saved_directory = output_directories[0]
    assert saved_directory.is_dir()
    assert {path.name for path in saved_directory.iterdir()} == {
        "HBV_output_2000-2001.png",
        "meteorological_data_2000-2001.png",
        "model_output_2000-2001.csv",
        "model_parameter.csv",
        "model_runoff_2000-2001.png",
        "model_stats_2000-2001.csv",
    }

    saved_output = pd.read_csv(
        saved_directory / "model_output_2000-2001.csv",
        index_col="TIMESTAMP",
        parse_dates=True,
    )
    assert_frame_equal(saved_output, output[1], check_exact=True, check_freq=False)
