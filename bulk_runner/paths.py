# bulk_runner/paths.py
from pathlib import Path
import os

def get_paths(
    outputs_env: str = "MATILDA_OUTPUTS",
    data_env: str = "MATILDA_DATA",
):
    """
    Returns a dict of canonical paths, anchored to the bulk_runner package.
    Works in scripts, PyCharm console, notebooks, anywhere.
    Environment variables can override:
      - MATILDA_OUTPUTS -> outputs directory
      - MATILDA_DATA    -> data directory
    """
    import bulk_runner as _br  # this module always has __file__
    bulk_dir   = Path(_br.__file__).resolve().parent
    repo_root  = bulk_dir.parent
    data_dir   = Path(os.getenv(data_env,   bulk_dir / "data")).resolve()
    outputs_dir= Path(os.getenv(outputs_env, bulk_dir / "outputs")).resolve()

    paths = {
        "repo_root": repo_root,
        "bulk_dir": bulk_dir,
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        # common files:
        "jobs": data_dir / "jobs.parquet",
        "catchments": data_dir / "catchments.parquet",
        "manifest": outputs_dir.parent / "runs_summary.parquet",  # as in your runner
    }
    return paths
