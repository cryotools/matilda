"""Synthetic smoke tests for an mspot-initiated model evaluation."""

from __future__ import annotations

import random

import numpy as np
from pandas.testing import assert_frame_equal, assert_index_equal
import pytest

from matilda.mspot_glacier import psample, spot_setup
from tests.synthetic import (
    AREA_CATCHMENT,
    SETUP_END,
    SETUP_START,
    SIMULATION_END,
    SIMULATION_START,
    calibration_parameters,
    make_synthetic_forcing,
    make_synthetic_glacier_profile,
    make_synthetic_mass_balance_observations,
    make_synthetic_observations,
)


pytestmark = pytest.mark.calibration


def test_mspot_model_evaluation_is_finite_and_has_no_file_side_effects(
    tmp_path,
    monkeypatch,
):
    forcing = make_synthetic_forcing()
    observations = make_synthetic_observations()
    forcing_before = forcing.copy(deep=True)
    observations_before = observations.copy(deep=True)
    monkeypatch.chdir(tmp_path)

    setup_class = spot_setup(
        set_up_start=SETUP_START,
        set_up_end=SETUP_END,
        sim_start=SIMULATION_START,
        sim_end=SIMULATION_END,
        freq="D",
        lat=45.0,
        area_cat=AREA_CATCHMENT,
        area_glac=20.0,
        ele_dat=1000.0,
        ele_cat=1200.0,
        ele_glac=1600.0,
        elev_rescaling=False,
    )
    setup = setup_class(forcing, observations, None)

    simulation = setup.simulation(calibration_parameters())
    evaluation = setup.evaluation()
    score = setup.objectivefunction(simulation, evaluation)

    assert_index_equal(
        simulation.index,
        evaluation.index,
        exact=True,
        check_names=False,
    )
    assert len(simulation) == 731
    assert np.isfinite(simulation).all()
    assert np.isfinite(evaluation).all()
    assert (simulation >= 0).all()
    assert np.isfinite(score)
    assert setup.objectivefunction(simulation, evaluation) == pytest.approx(score)
    assert_frame_equal(forcing, forcing_before, check_exact=True)
    assert_frame_equal(observations, observations_before, check_exact=True)
    assert list(tmp_path.iterdir()) == []


def test_glacier_only_psample_completes_one_evaluation(tmp_path):
    numpy_random_state = np.random.get_state()
    python_random_state = random.getstate()
    forcing = make_synthetic_forcing()
    observations = make_synthetic_mass_balance_observations()
    profile = make_synthetic_glacier_profile()

    try:
        np.random.seed(0)
        results = psample(
            forcing,
            observations,
            rep=1,
            output=tmp_path,
            dbname="glacier_only_smoke",
            dbformat="ram",
            set_up_start=SETUP_START,
            set_up_end=SETUP_END,
            sim_start=SIMULATION_START,
            sim_end=SIMULATION_END,
            freq="D",
            lat=45.0,
            area_cat=AREA_CATCHMENT,
            area_glac=20.0,
            ele_dat=1000.0,
            ele_glac=1600.0,
            glacier_profile=profile,
            glacier_only=True,
            obs_type="annual",
            algorithm="lhs",
            obj_dir="minimize",
            save_sim=False,
        )
    finally:
        np.random.set_state(numpy_random_state)
        random.setstate(python_random_state)

    assert results["best_index"] == 0
    assert np.isfinite(results["best_objf"])
    assert (tmp_path / "glacier_only_smoke_observations.csv").is_file()
