from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from typing import Any
import pandas as pd
import traceback
import json, yaml, os, time, hashlib, contextlib

from .schemas import MatildaJob

# ---------- helpers ----------
def _read_any_settings(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f) or {}
        return json.load(f)

def _read_params(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _read_forcing(path: str) -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Forcing file not found: {p}")

    # Keep names as-is; just trim/strip and handle BOM safely
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    if "TIMESTAMP" not in df.columns:
        raise KeyError(f"No TIMESTAMP column in {p}. Columns read: {list(df.columns)}")

    # Parse to datetime
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="raise")
    return df


def _read_glacier_profile(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    return pd.read_csv(path)

def _expected_outputs(base: Path) -> list[Path]:
    return [base / "discharge.parquet", base / "meta.parquet"]

def _all_exist(paths: list[Path]) -> bool:
    return all(p.exists() for p in paths)

def _hash_row(d: dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha256(s).hexdigest()[:16]

# ---------- stateless worker ----------
def run_matilda_job(job: MatildaJob) -> dict[str, Any]:
    # Import MATILDA
    from matilda.core import matilda_simulation

    base = Path(job.out_dir) / job.catchment_id / job.scenario / job.model / job.run_id
    base.mkdir(parents=True, exist_ok=True)

    if _all_exist(_expected_outputs(base)):
        return {
            "run_id": job.run_id, "catchment_id": job.catchment_id, "parent_id": job.parent_id,
            "scenario": job.scenario, "model": job.model, "result_path": str(base),
            "ok": True, "skipped": True, "error": None,
        }

    started = time.time()
    try:
        forcing  = _read_forcing(job.forcing_path)
        params   = _read_params(job.params_path)
        settings = _read_any_settings(job.settings_path)

        glac = _read_glacier_profile(job.glacier_profile_path)
        if glac is not None:
            # Use the keyword your matilda_simulation expects for a glacier profile:
            settings["glacier_profile"] = glac

        # Run quietly (better logs under parallel)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            out = matilda_simulation(forcing, **settings, **params)

        # Retrieve key outputs
        model_output = out[0]
        glacier_rescaling = out[5]

        # Save outputs
        base.mkdir(parents=True, exist_ok=True)

        # Prefer pyarrow if available
        PARQUET_ENGINE = "pyarrow"

        # 1) discharge / main output
        model_output.to_parquet(base / "discharge.parquet", engine=PARQUET_ENGINE)

        # 2) glacier rescaling
        glacier_rescaling.to_parquet(base / "glacier_rescaling.parquet", engine=PARQUET_ENGINE)

        # tiny meta for quick scans (optional but handy)
        pd.DataFrame([{
            "ok": True,
            "rows_discharge": len(model_output),
            "cols_discharge": model_output.shape[1],
            "rows_glacier_rescaling": len(glacier_rescaling),
            "cols_glacier_rescaling": glacier_rescaling.shape[1],
        }]).to_parquet(base / "meta.parquet", engine=PARQUET_ENGINE)

    except Exception as e:
        ok, err = False, repr(e)
        # write full traceback into the run folder for quick debugging
        base = Path(job.out_dir) / job.catchment_id / job.scenario / job.model / job.run_id
        base.mkdir(parents=True, exist_ok=True)
        (base / "error.log").write_text(traceback.format_exc())

    finished = time.time()
    manifest = {
        "run_id": job.run_id,
        "catchment_id": job.catchment_id,
        "parent_id": job.parent_id,
        "scenario": job.scenario,
        "model": job.model,
        "result_path": str(base),
        "ok": ok,
        "error": err,
        "started_at": pd.Timestamp.utcfromtimestamp(started),
        "finished_at": pd.Timestamp.utcfromtimestamp(finished),
        "job_hash": _hash_row(asdict(job)),
        "params_hash": _hash_row(_read_params(job.params_path)) if ok else None,
        "settings_hash": _hash_row(_read_any_settings(job.settings_path)) if ok else None,
    }
    pd.DataFrame([manifest]).to_parquet(base / "run_manifest.parquet")
    return manifest
