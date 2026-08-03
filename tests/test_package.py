"""Package metadata and resource checks."""

from __future__ import annotations

import ast
from importlib import metadata, resources
import json
from pathlib import Path

import yaml

import matilda


PROJECT_ROOT = Path(__file__).parents[1]


def _assigned_string(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {path}")


def test_version_metadata_is_consistent():
    expected = "1.0.2"
    citation = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text())
    zenodo = json.loads((PROJECT_ROOT / ".zenodo.json").read_text())

    assert matilda.__version__ == expected
    assert metadata.version("matilda") == expected
    assert _assigned_string(PROJECT_ROOT / "docs/conf.py", "release") == expected
    assert citation["version"] == expected
    assert zenodo["version"] == expected


def test_parameter_resource_has_documented_defaults():
    resource = resources.files("matilda").joinpath("parameters.json")
    parameters = json.loads(resource.read_text(encoding="utf-8"))["parameters"]

    assert len(parameters) == 23
    assert parameters["CFMAX_snow"]["default"] == 2.5
    assert parameters["hydro_year"]["default"] == 10
    assert parameters["pfilter"]["default"] == 0


def test_public_modules_are_importable():
    assert matilda.core.__name__ == "matilda.core"
    assert matilda.mspot_glacier.__name__ == "matilda.mspot_glacier"
