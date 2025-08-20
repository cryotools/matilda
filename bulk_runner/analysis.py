from pathlib import Path
import pandas as pd

man = pd.read_parquet("bulk_runner/runs_summary.parquet")
cat = pd.read_parquet("bulk_runner/data/catchments.parquet")  # if you keep one
df = man.merge(cat[["catchment_id", "glacierized"]], on="catchment_id", how="left")

# Example filters
calib = df.query("ok and scenario == 'calib'")
ssp2  = df.query("ok and scenario == 'SSP2' and model == 'GFDL'")

# Loader helper (from 6))
def load_runs(manifest: pd.DataFrame, rows: pd.DataFrame, kind="discharge"):
    from pathlib import Path
    files = []
    for r in rows.itertuples(index=False):
        p = Path(r.result_path) / f"{kind}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d["run_id"] = r.run_id
            d["catchment_id"] = r.catchment_id
            d["scenario"] = r.scenario
            d["model"] = r.model
            files.append(d)
    return pd.concat(files) if files else pd.DataFrame()

Q_calib = load_runs(man, calib, kind="discharge")
Q_ssp2  = load_runs(man, ssp2,  kind="discharge")

print(Q_calib.head())
print(Q_ssp2.head())
