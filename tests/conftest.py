"""Shared fixtures and command-line options for MATILDA tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matilda-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "matilda-cache")
)

from tests.scenario import MANIFEST_PATH, load_reference, run_reference_scenario


def pytest_addoption(parser):
    parser.addoption(
        "--impact-report",
        action="store",
        default=None,
        metavar="DIRECTORY",
        help="write per-variable and annual model-impact CSV reports",
    )


@pytest.fixture(scope="session")
def reference_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def reference_output():
    return load_reference()


@pytest.fixture(scope="session")
def current_output():
    return run_reference_scenario()


@pytest.fixture(scope="session")
def impact_report_directory(pytestconfig):
    value = pytestconfig.getoption("--impact-report")
    return Path(value).resolve() if value else None
