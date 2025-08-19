from .orchestrator import run_jobs
import pandas as pd

def main():
    manifest = run_jobs("bulk_runner/data/jobs.parquet", max_workers=None)
    print(manifest[["run_id","catchment_id","scenario","model","ok","result_path","error"]])

    # Optional: join with a catchment registry for filtering
    reg_path = "bulk_runner/data/catchments.parquet"
    try:
        cat = pd.read_parquet(reg_path)
        mf = manifest.merge(cat[["catchment_id","glacierized","glacier_cover_frac"]],
                            on="catchment_id", how="left")
        print("\nSuccessful runs for glacierized catchments:")
        print(mf.query("ok and glacierized")[["run_id","catchment_id","scenario","model","result_path"]])
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
