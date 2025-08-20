from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MatildaJob:
    run_id: str
    catchment_id: str
    parent_id: str
    scenario: str
    model: str
    forcing_path: str                 # required
    params_path: Optional[str] = None # optional parameters.yml/.json
    obs_path: Optional[str] = None    # optional gauging/observations CSV
    settings_path: Optional[str] = None
    glacier_profile_path: Optional[str] = None
    out_dir: str = "bulk_runner/outputs"