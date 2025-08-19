from dataclasses import dataclass

@dataclass(frozen=True)
class MatildaJob:
    run_id: str
    catchment_id: str
    parent_id: str
    scenario: str
    model: str
    forcing_path: str               # ERA5L CSV for THIS run
    params_path: str                # parameters.yml / .json
    settings_path: str | None       # optional .yml/.json
    glacier_profile_path: str | None  # only if glacierized
    out_dir: str                    # e.g., "bulk_runner/outputs"
