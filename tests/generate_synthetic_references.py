"""Generate maintained references for deterministic synthetic model modes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
import pickle
import platform

import matplotlib.pyplot as plt

from matilda.core import matilda_simulation
from tests.synthetic import (
    REFERENCE_DIRECTORY,
    REFERENCE_MANIFEST_PATH,
    SETUP_END,
    SETUP_START,
    SIMULATION_END,
    SIMULATION_START,
    make_synthetic_forcing,
    model_settings,
)


REFERENCE_MODES = {"zero_glacier": 0.0, "fixed_glacier": 20.0}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace existing references after an accepted scientific change",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = [
        REFERENCE_DIRECTORY / f"synthetic_{name}_output.pickle"
        for name in REFERENCE_MODES
    ]
    paths.append(REFERENCE_MANIFEST_PATH)
    existing = [path for path in paths if path.exists()]
    if existing and not args.replace:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"Refusing to replace maintained references without --replace: {names}"
        )

    references = {}
    for name, area_glac in REFERENCE_MODES.items():
        try:
            with redirect_stdout(io.StringIO()):
                output = matilda_simulation(
                    make_synthetic_forcing(),
                    **model_settings(area_glac),
                )
        finally:
            plt.close("all")
        reference_path = REFERENCE_DIRECTORY / f"synthetic_{name}_output.pickle"
        reference_data = pickle.dumps(output, protocol=5)
        reference_path.write_bytes(reference_data)
        references[name] = {
            "file": reference_path.name,
            "sha256": sha256(reference_data).hexdigest(),
            "area_glac_km2": area_glac,
        }

    manifest = {
        "schema_version": 1,
        "core_baseline_commit": "7a7625c9365e21dfe1b9e267b08139774e37c446",
        "model_version": "1.0.2",
        "python": platform.python_version(),
        "setup_period": f"{SETUP_START}/{SETUP_END}",
        "simulation_period": f"{SIMULATION_START}/{SIMULATION_END}",
        "comparison_policy": (
            "Exact values, structure, dtypes, indexes, columns, and "
            "missing-value positions"
        ),
        "references": references,
    }
    REFERENCE_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
