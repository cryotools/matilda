from pathlib import Path
import pandas as pd

def main():
    Path("bulk_runner/data").mkdir(parents=True, exist_ok=True)
    Path("bulk_runner/outputs").mkdir(parents=True, exist_ok=True)

    # Catchment registry (facts for filtering)
    catchments = pd.DataFrame([
        dict(catchment_id="TEST_GLAC", parent_id="TEST_GLAC", name="Glacier Test",
             area_km2=120.0, glacierized=True, glacier_area_km2=22.0, glacier_cover_frac=0.183),
        dict(catchment_id="TEST_NONGR", parent_id="TEST_NONGR", name="Non‑glacier Test",
             area_km2=95.0, glacierized=False, glacier_area_km2=0.0, glacier_cover_frac=0.0),
    ])
    catchments.to_parquet("bulk_runner/data/catchments.parquet", index=False)

    # Job table (one row = one run)
    jobs = pd.DataFrame([
        dict(run_id="run_001",
             catchment_id="TEST_GLAC", parent_id="TEST_GLAC",
             scenario="SSP2", model="GFDL",
             forcing_path="tests/test_input/era5.csv",
             params_path="tests/test_input/parameters.yml",
             settings_path="tests/test_input/settings.yml",
             glacier_profile_path="tests/test_input/glacier_profile.csv",
             out_dir="bulk_runner/outputs"),
        dict(run_id="run_002",
             catchment_id="TEST_NONGR", parent_id="TEST_NONGR",
             scenario="SSP5", model="MPI-ESM",
             forcing_path="tests/test_input/era5.csv",
             params_path="tests/test_input/parameters.yml",
             settings_path="tests/test_input/settings.yml",
             glacier_profile_path=None,
             out_dir="bulk_runner/outputs"),
    ])
    jobs.to_parquet("bulk_runner/data/jobs.parquet", index=False)
    print("Wrote bulk_runner/data/{catchments, jobs}.parquet")

if __name__ == "__main__":
    main()
