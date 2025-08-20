from pathlib import Path
import sys
import pandas as pd

# --- set this when running in a console/IDE (ignored if CLI arg is given) ---
bulk_root = '/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/bulk_runner'  # e.g. bulk_root = "/home/you/path/to/repo/bulk_runner"

# --- CLI override ---
# if len(sys.argv) > 1:
#     bulk_root = sys.argv[1]

if not bulk_root:
    if Path('bulk_runner').is_dir():
        bulk_root = Path('bulk_runner')
    else:
        raise SystemExit("Please pass / set the absolute path to your bulk_runner directory.")

BULK = Path(bulk_root).expanduser().resolve()
MANIFEST = BULK / "runs_summary.parquet"
CATCHMENTS = BULK / "data" / "catchments.parquet"

if not MANIFEST.exists():
    raise FileNotFoundError(f"runs_summary not found at: {MANIFEST}")

print(f"Using bulk_root: {BULK}")

# --- load manifest ---
man = pd.read_parquet(MANIFEST)

# --- load catchment registry if present (optional) ---
if CATCHMENTS.exists():
    cat = pd.read_parquet(CATCHMENTS)
    # keep only the few columns we need; adjust as you like
    keep_cols = [c for c in ["catchment_id", "glacierized", "glacier_cover_frac"] if c in cat.columns]
    df = man.merge(cat[keep_cols], on="catchment_id", how="left")
else:
    print(f"(Info) No catchments registry at {CATCHMENTS}; proceeding without it.")
    df = man.copy()

# --- example filters ---
calib = df.query("ok and scenario == 'calib'")
ssp2  = df.query("ok and scenario == 'SSP2' and model == 'GFDL'")

# --- helper to load many run files of a given kind (e.g., 'discharge' / 'glacier_rescaling') ---
def load_runs(manifest: pd.DataFrame, rows: pd.DataFrame, kind: str = "discharge") -> pd.DataFrame:
    files = []
    for r in rows.itertuples(index=False):
        p = Path(r.result_path) / f"{kind}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            # add identifiers to each loaded table for stacking/plotting later
            d["run_id"] = r.run_id
            d["catchment_id"] = r.catchment_id
            d["scenario"] = r.scenario
            d["model"] = r.model
            files.append(d)
    return pd.concat(files, ignore_index=False) if files else pd.DataFrame()

# --- load examples ---
Q_calib = load_runs(man, calib, kind="discharge")
Q_ssp2  = load_runs(man, ssp2,  kind="discharge")

print("\nCalib sample:")
print(Q_calib.head())
print("\nSSP2 (GFDL) sample:")
print(Q_ssp2.head())
