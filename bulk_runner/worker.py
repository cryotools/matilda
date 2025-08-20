from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from typing import Any
import pandas as pd
import json, yaml, os, time, hashlib, contextlib, warnings, traceback, sys

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

    # Deterministic per-run output dir
    scenario = job.scenario.strip()
    model = job.model.strip()
    base = Path(job.out_dir) / job.catchment_id / scenario / model / job.run_id
    base.mkdir(parents=True, exist_ok=True)

    # Prepare log file and redirect stdout/stderr
    log_path = base / "console.log"

    # Read inputs OUTSIDE the redirection so file existence errors show up in console.log too
    try:
        forcing  = _read_forcing(job.forcing_path)
        params   = _read_params(job.params_path)
        settings = _read_any_settings(job.settings_path)
        glac     = _read_glacier_profile(job.glacier_profile_path)
        if glac is not None:
            settings["glacier_profile"] = glac
    except Exception:
        (base / "error.log").write_text(traceback.format_exc())
        return {
            "run_id": job.run_id, "catchment_id": job.catchment_id, "parent_id": job.parent_id,
            "scenario": scenario, "model": model, "result_path": str(base),
            "ok": False, "error": "Input loading failed; see error.log",
        }

    # Redirect both stdout & stderr to the per-run log file
    with open(log_path, "w") as logf, contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
        # Ensure Python warnings are emitted (not silenced)
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("default")  # show once per location

            print(f"[RUN START] {pd.Timestamp.utcnow().isoformat()}Z")
            print(f"[JOB] {asdict(job)}")
            print(f"[INFO] Forcing shape: {forcing.shape}, columns: {list(forcing.columns)}")
            print(f"[INFO] Params keys: {list(params.keys())}")
            print(f"[INFO] Settings keys: {list(settings.keys())}")

            started = time.time()
            ok = False
            err = None
            model_output = None
            glacier_rescaling = None

            try:
                # Run MATILDA (no devnull; everything goes into console.log)
                out = matilda_simulation(forcing, **settings, **params)

                # Adapt if your function returns differently
                model_output = out[0]
                glacier_rescaling = out[5]

                # --- Save outputs (simple, DataFrame case) ---
                PARQUET_ENGINE = "pyarrow"
                (base / "discharge.parquet").unlink(missing_ok=True)
                (base / "glacier_rescaling.parquet").unlink(missing_ok=True)

                if isinstance(model_output, pd.DataFrame):
                    model_output.to_parquet(base / "discharge.parquet", engine=PARQUET_ENGINE)
                else:
                    print(f"[WARN] model_output is {type(model_output)}; writing repr()")
                    (base / "discharge.txt").write_text(repr(model_output))

                if isinstance(glacier_rescaling, pd.DataFrame):
                    glacier_rescaling.to_parquet(base / "glacier_rescaling.parquet", engine=PARQUET_ENGINE)
                else:
                    print(f"[WARN] glacier_rescaling is {type(glacier_rescaling)}; writing repr()")
                    (base / "glacier_rescaling.txt").write_text(repr(glacier_rescaling))

                # Tiny meta
                meta = [{
                    "ok": True,
                    "rows_discharge": getattr(model_output, "shape", (None, None))[0],
                    "cols_discharge": getattr(model_output, "shape", (None, None))[1],
                    "rows_glacier_rescaling": getattr(glacier_rescaling, "shape", (None, None))[0],
                    "cols_glacier_rescaling": getattr(glacier_rescaling, "shape", (None, None))[1],
                }]
                pd.DataFrame(meta).to_parquet(base / "meta.parquet", engine=PARQUET_ENGINE)

                ok = True

            except Exception as e:
                err = repr(e)
                # Write full traceback for this run
                (base / "error.log").write_text(traceback.format_exc())
                print("[ERROR] Exception during run:", err)

            finally:
                finished = time.time()
                print(f"[RUN END] {pd.Timestamp.utcnow().isoformat()}Z")
                print(f"[DURATION] {finished - started:.2f} s")
                logf.flush()

    # Return manifest row
    return {
        "run_id": job.run_id,
        "catchment_id": job.catchment_id,
        "parent_id": job.parent_id,
        "scenario": scenario,
        "model": model,
        "result_path": str(base),
        "ok": ok,
        "error": err,
        "started_at": pd.Timestamp.utcfromtimestamp(started) if 'started' in locals() else None,
        "finished_at": pd.Timestamp.utcfromtimestamp(finished) if 'finished' in locals() else None,
    }