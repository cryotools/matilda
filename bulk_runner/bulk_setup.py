import pandas as pd
from pathlib import Path

jobs = pd.DataFrame([
    # Calibration
    dict(run_id="run_001",
         catchment_id="kyzylsuu", parent_id="kyzylsuu",
         scenario="calib", model="ERA5L", run_type="calib", period="1981-2010",
         forcing_path="/abs/path/kyzylsuu_era5l.csv",
         params_path="/abs/path/params_calib.yml",
         settings_path="/abs/path/settings_calib.yml",
         glacier_profile_path="/abs/path/glacier_profile.csv",
         out_dir="bulk_runner/outputs"),
    # Projection SSP2 with GFDL
    dict(run_id="run_002",
         catchment_id="kyzylsuu", parent_id="kyzylsuu",
         scenario="SSP2", model="GFDL", run_type="projection", period="2015-2100",
         forcing_path="/abs/path/kyzylsuu_ssp2_gfdl.csv",
         params_path="/abs/path/params_ssp2.yml",
         settings_path="/abs/path/settings_ssp2.yml",
         glacier_profile_path="/abs/path/glacier_profile.csv",
         out_dir="bulk_runner/outputs"),
])
# Sanity checks
assert jobs["run_id"].is_unique, "run_id must be unique"
for col in ["forcing_path","params_path"]:
    assert jobs[col].map(lambda p: Path(p).exists()).all(), f"missing files in {col}"

jobs.to_parquet("bulk_runner/data/jobs.parquet", index=False)
