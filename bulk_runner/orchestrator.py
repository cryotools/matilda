from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pandas as pd
from tqdm import tqdm

from .schemas import MatildaJob
from .worker import run_matilda_job

def _row_to_job(r) -> MatildaJob:
    return MatildaJob(
        run_id=r.run_id, catchment_id=r.catchment_id, parent_id=r.parent_id,
        scenario=r.scenario, model=r.model,
        forcing_path=r.forcing_path, params_path=r.params_path,
        settings_path=getattr(r, "settings_path", None),
        glacier_profile_path=getattr(r, "glacier_profile_path", None),
        out_dir=r.out_dir,
    )

def run_jobs(job_table_path: str, max_workers: int | None = None) -> pd.DataFrame:
    jobs_df = pd.read_parquet(job_table_path) if job_table_path.endswith(".parquet") else pd.read_csv(job_table_path)
    jobs = [_row_to_job(r) for r in jobs_df.itertuples(index=False)]
    max_workers = max_workers or max(os.cpu_count() - 1, 1)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_matilda_job, j): j for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="MATILDA runs"):
            j = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "run_id": j.run_id, "catchment_id": j.catchment_id, "parent_id": j.parent_id,
                    "scenario": j.scenario, "model": j.model, "result_path": None,
                    "ok": False, "error": repr(e),
                }
            results.append(res)

    manifest = pd.DataFrame(results)
    # Save a global manifest next to the local outputs dir
    outdir = jobs_df["out_dir"].iloc[0]
    manifest.to_parquet(os.path.join(outdir, "..", "runs_summary.parquet"), index=False)
    return manifest
