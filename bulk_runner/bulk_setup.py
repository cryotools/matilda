from pathlib import Path
from typing import Callable, Iterable, Optional, Union
import yaml
import re
import pandas as pd

## Edit parameters of multiple catchments


def _safe_eval(expr: str, settings: dict) -> bool:
    """
    Evaluate a simple boolean expression against a settings dict.
    Allows names, numbers, booleans, None, and operators: < <= > >= == != and or not ().
    Example: "lat > 42.18 and area_cat < 200"
    Missing keys resolve to None.
    """
    # allowed tokens pattern (rough sanity)
    if not re.fullmatch(r"[A-Za-z0-9_().<>=!\-\+\*/\s'\"andornot]+", expr):
        raise ValueError("Expression contains unsupported characters.")

    local = {k: settings.get(k, None) for k in settings.keys()}
    # also allow lowercase aliases (in case settings has mixed case)
    local.update({k.lower(): v for k, v in settings.items() if k.lower() not in local})

    # very small safe builtins
    safe_builtins = {"True": True, "False": False, "None": None}
    try:
        return bool(eval(expr, {"__builtins__": {}}, {**safe_builtins, **local}))
    except Exception:
        return False


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def _write_yaml(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def edit_parameters(
    catchments_root: Union[str, Path],
    *,
    updates: dict,
    catchment_names: Optional[Iterable[str]] = None,   # if None -> all subdirs
    where: Optional[Union[str, Callable[[dict, str, Path], bool]]] = None,
    params_filename: str = "parameters.yml",
    settings_filename: str = "settings.yml",
    create_if_missing: bool = False,
    template_path: Optional[Union[str, Path]] = None,  # used only if creating a new file
    backup: bool = True,
    dry_run: bool = False,
) -> list[Path]:
    """
    Update selected keys in parameters.yml across catchment folders.

    - updates: dict of key -> new value (only these keys are changed; others preserved)
    - catchment_names: optional list of folder names to include; None = all
    - where:
        * None: no extra filter
        * str: boolean expression evaluated on settings.yml, e.g. "lat > 42.18 and area_cat < 200"
        * callable: lambda settings, catchment_name, catchment_path -> bool
    - create_if_missing: if True, creates parameters.yml when absent (optionally from template_path)
    - template_path: optional YAML template used only when creating new parameters.yml
    - backup: if True, writes parameters.yml.bak before saving changes
    - dry_run: if True, prints what would change but doesn’t write

    Returns list of paths actually written.
    """
    root = Path(catchments_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    if not isinstance(updates, dict) or not updates:
        raise ValueError("`updates` must be a non-empty dict of {param: value}.")

    names_set = set(catchment_names) if catchment_names is not None else None
    written: list[Path] = []

    for catch_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        name = catch_dir.name
        if names_set is not None and name not in names_set:
            continue

        settings_path = catch_dir / settings_filename
        if not settings_path.exists():
            print(f"[SKIP] {name}: no {settings_filename}")
            continue

        try:
            settings = _read_yaml(settings_path)
        except Exception as e:
            print(f"[SKIP] {name}: cannot read {settings_filename}: {e}")
            continue

        # Apply condition
        allowed = True
        if where is not None:
            if isinstance(where, str):
                allowed = _safe_eval(where, settings)
            elif callable(where):
                try:
                    allowed = bool(where(settings, name, catch_dir))
                except Exception as e:
                    print(f"[SKIP] {name}: where() raised {e}")
                    allowed = False
            else:
                raise TypeError("`where` must be None, a str expression, or a callable.")
        if not allowed:
            continue

        params_path = catch_dir / params_filename
        if params_path.exists():
            try:
                params = _read_yaml(params_path)
            except Exception as e:
                print(f"[SKIP] {name}: cannot read {params_filename}: {e}")
                continue
        else:
            if not create_if_missing:
                print(f"[SKIP] {name}: no {params_filename} (creation disabled)")
                continue
            if template_path:
                t = Path(template_path).expanduser().resolve()
                if not t.exists():
                    print(f"[SKIP] {name}: template not found: {t}")
                    continue
                try:
                    params = _read_yaml(t)
                    print(f"[INFO] {name}: creating {params_filename} from template")
                except Exception as e:
                    print(f"[SKIP] {name}: cannot read template: {e}")
                    continue
            else:
                params = {}
                print(f"[INFO] {name}: creating empty {params_filename}")

        # Apply only the requested keys
        before = dict(params)
        params.update(updates)

        if params == before:
            print(f"[NO-OP] {name}: no changes")
            continue

        print(f"[CHANGE] {name}: { {k: (before.get(k), params.get(k)) for k in updates.keys()} }")

        if dry_run:
            continue

        try:
            if backup and params_path.exists():
                params_path.with_suffix(params_path.suffix + ".bak").write_text(
                    yaml.safe_dump(before, sort_keys=False))
            _write_yaml(params_path, params)
            written.append(params_path)
        except Exception as e:
            print(f"[ERROR] {name}: failed to write {params_filename}: {e}")

    print(f"[DONE] Edited {len(written)} catchments.")
    return written

## Example usage:
#
# # Change one parameter in all catchments:
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"PCORR": 0.35},
#     where=None,               # apply to all
#     create_if_missing=False,  # don’t create new files, only edit existing ones
# )
#
# # Change one parameter in specific catchments with two conditions:
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"TT_snow": -1.5},
#     where="lat > 42.18 and area_cat < 200",
# )
#
# # Create new parameters.yml in all catchments that lack it, from a template:
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"TT_snow": -1.5},
#     where=None,  # all
#     create_if_missing=True,
#     template_path="/abs/path/to/param_template.yml",
# )
#
# # Custom predicate function:
# def pred(settings, name, path):
#     return name.startswith("kyz") and float(settings.get("area_cat", 1e9)) < 300
#
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"TT_snow": -1.5},
#     where=pred,
# )
#
# # Dry-run first:
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"TT_snow": -1.5},
#     where="region == 'TienShan' or lat >= 42",
#     dry_run=True,   # show planned changes only
# )
# # If happy:
# edit_parameters(
#     "/abs/path/to/catchments",
#     updates={"TT_snow": -1.5},
#     where="region == 'TienShan' or lat >= 42",
#     dry_run=False,
# )

## Issyk Kul setup example:
ik_path = Path('/home/phillip/Seafile/EBA-CA/Papers/No3_Issyk-Kul/calibration/calibration_data')

edit_parameters(
    catchments_root=ik_path,
    updates={"TT_diff": 0.76, "TT_snow": -1.445},
    create_if_missing=True,
    template_path="/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/parameters.yml",
    dry_run=False
)
## Functions to build jobs from catchment directories

# Create catchments registry
def build_or_update_catchment_registry(
    catchments_root: str | Path,
    bulk_runner_root: str | Path,
    *,
    out_path_rel: str = "bulk_runner/data/catchments.parquet",
    settings_filename: str = "settings.yml",
    glacier_profile_filename: str = "glacier_profile.csv",
    extra_columns: Optional[Iterable[str]] = None,  # keep/merge any manual columns
) -> pd.DataFrame:
    """
    Scan catchment directories and (re)write a registry parquet with useful fields.

    Columns created:
      - catchment_id
      - parent_id           (from settings.yml 'parent_id' if present, else catchment_id)
      - name                (settings.yml 'name' else catchment_id)
      - area_km2            (settings.yml 'area_cat' if present)
      - lat, lon            (from settings.yml if present)
      - glacierized         (True if glacier_profile.csv exists OR settings.yml 'glacierized'==True)
      - glacier_area_km2    (settings.yml if present)
      - glacier_cover_frac  (settings.yml if present)
      - root_path           (absolute path to catchment folder)
    If an existing registry exists, we keep/merge any columns listed in `extra_columns`.
    """
    catch_root = Path(catchments_root).expanduser().resolve()
    bulk_root = Path(bulk_runner_root).expanduser().resolve()
    out_path = (bulk_root / out_path_rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prior registry (optional)
    prev = None
    if out_path.exists():
        prev = pd.read_parquet(out_path)

    rows = []
    for cd in sorted(p for p in catch_root.iterdir() if p.is_dir()):
        name = cd.name
        settings_path = cd / settings_filename
        glacier_path = cd / glacier_profile_filename

        cfg = {}
        if settings_path.exists():
            try:
                cfg = yaml.safe_load(settings_path.read_text()) or {}
            except Exception:
                cfg = {}

        parent_id = cfg.get("parent_id", name)
        area = cfg.get("area_cat", None)
        lat = cfg.get("lat", None)
        lon = cfg.get("lon", None)
        glacier_area = cfg.get("glacier_area_km2", None)
        glacier_frac = cfg.get("glacier_cover_frac", None)

        glacierized = bool(cfg.get("glacierized", False)) or glacier_path.exists()

        rows.append(dict(
            catchment_id=name,
            parent_id=parent_id,
            name=cfg.get("name", name),
            area_km2=area,
            lat=lat,
            lon=lon,
            glacierized=glacierized,
            glacier_area_km2=glacier_area,
            glacier_cover_frac=glacier_frac,
            root_path=str(cd.resolve()),
        ))

    reg = pd.DataFrame(rows)

    # Merge with previous to preserve extra/manual columns if requested
    if prev is not None and extra_columns:
        keep = ["catchment_id"] + [c for c in extra_columns if c in prev.columns and c != "catchment_id"]
        if len(keep) > 1:
            reg = reg.merge(prev[keep], on="catchment_id", how="left")

    reg.to_parquet(out_path, index=False)
    print(f"[OK] catchment registry written: {out_path} (rows={len(reg)})")
    return reg

# Handle run_id numbering
def _extract_max_run_no_from_strings(vals) -> int:
    """Find max run_### number in an iterable of strings."""
    pat = re.compile(r"run_(\d{1,})$")
    m = -1
    for v in vals:
        if not isinstance(v, str):
            continue
        mo = pat.search(v.strip())
        if mo:
            try:
                m = max(m, int(mo.group(1)))
            except ValueError:
                pass
    return m

def _discover_max_run_no_in_outputs(outputs_root: Path) -> int:
    """
    Walk bulk_runner/outputs and find deepest run_* dirs:
    outputs/<catchment>/<scenario>/<model>/<run_id>/
    """
    if not outputs_root.exists():
        return -1
    pat = re.compile(r"run_(\d{1,})$")
    m = -1
    # Only look 4 levels deep (catchment/scenario/model/run_id)
    for catch_dir in outputs_root.iterdir():
        if not catch_dir.is_dir():
            continue
        for scen_dir in catch_dir.iterdir():
            if not scen_dir.is_dir():
                continue
            for model_dir in scen_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for run_dir in model_dir.iterdir():
                    if run_dir.is_dir():
                        mo = pat.fullmatch(run_dir.name)
                        if mo:
                            try:
                                m = max(m, int(mo.group(1)))
                            except ValueError:
                                pass
    return m

# Job builder
def build_jobs_from_catchments(
    catchments_root: str | Path,
    bulk_runner_root: str | Path,
    *,
    scenario: str = "calib",
    model: str = "CHELSA",
    forcing_filename: str = "chelsa.csv",
    glacier_profile_filename: str = "glacier_profile.csv",
    settings_filename: str = "settings.yml",
    gauging_filename: str = "gauging.csv",       # optional; if missing, left None
    params_filename: str = "parameters.yml",     # optional; if missing, can fall back to global
    global_params_path: Optional[str | Path] = None,  # used if a catchment lacks its own parameters.yml
    out_dir_rel: str = "bulk_runner/outputs",
    jobs_path_rel: str = "bulk_runner/data/jobs.parquet",
    parent_id_mode: str = "self",  # "self" or "from_settings_key"
    parent_id_settings_key: str = "parent_id",   # used if parent_id_mode == "from_settings_key"
    run_id_prefix: str = "run_",
    append: bool = False,  # if True, append to an existing jobs.parquet (and avoid duplicates by run_id)
) -> pd.DataFrame:
    """
    Option B: Assign ONE new run_id to all rows created in this call.
    The new id is next after the max observed in jobs.parquet and outputs/*/*/*/run_*.
    """
    catchments_root = Path(catchments_root).expanduser().resolve()
    bulk_runner_root = Path(bulk_runner_root).expanduser().resolve()

    out_dir = (bulk_runner_root / out_dir_rel).resolve()
    jobs_path = (bulk_runner_root / jobs_path_rel).resolve()
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing (if append)
    existing = pd.DataFrame()
    if append and jobs_path.exists():
        existing = pd.read_parquet(jobs_path)

    rows = []
    for catch_dir in sorted([p for p in catchments_root.iterdir() if p.is_dir()]):
        catchment_id = catch_dir.name

        forcing_path = catch_dir / forcing_filename
        settings_path = catch_dir / settings_filename
        glacier_path = catch_dir / glacier_profile_filename
        gauging_path = catch_dir / gauging_filename
        params_path = catch_dir / params_filename

        # Required files check
        if not forcing_path.exists():
            print(f"[SKIP] {catchment_id}: missing {forcing_filename}")
            continue
        if not settings_path.exists():
            print(f"[SKIP] {catchment_id}: missing {settings_filename}")
            continue

        # Parent id
        if parent_id_mode == "self":
            parent_id = catchment_id
        else:
            try:
                _cfg = yaml.safe_load(settings_path.read_text()) or {}
                parent_id = _cfg.get(parent_id_settings_key, catchment_id)
            except Exception:
                parent_id = catchment_id

        # Parameters path: prefer local; else global; else None
        if params_path.exists():
            final_params_path = str(params_path)
        elif global_params_path is not None:
            final_params_path = str(Path(global_params_path).expanduser().resolve())
        else:
            final_params_path = None

        # Glacier profile optional; obs (gauging) optional
        final_glacier_path = str(glacier_path) if glacier_path.exists() else None
        final_obs_path = str(gauging_path) if gauging_path.exists() else None

        # Create row with blank run_id (to be filled if it's a new row)
        row = dict(
            run_id="",  # will be filled with the batch id below
            catchment_id=catchment_id,
            parent_id=parent_id,
            scenario=scenario.strip(),
            model=model.strip(),
            forcing_path=str(forcing_path),
            params_path=final_params_path,
            settings_path=str(settings_path),
            glacier_profile_path=final_glacier_path,
            obs_path=final_obs_path,
            out_dir=str(out_dir),
        )
        rows.append(row)

    new = pd.DataFrame(rows)

    if new.empty:
        if append and not existing.empty:
            print(f"[OK] nothing new; keeping existing: {jobs_path} (rows={len(existing)})")
            return existing
        print("[WARN] no catchment subdirs found.")
        return pd.DataFrame()

    # Determine next global run number (Option B)
    max_from_jobs = -1
    if jobs_path.exists():
        try:
            jp = pd.read_parquet(jobs_path)
            max_from_jobs = _extract_max_run_no_from_strings(jp.get("run_id", []))
        except Exception:
            pass

    max_from_fs = _discover_max_run_no_in_outputs(out_dir)

    current_max = max(max_from_jobs, max_from_fs)
    next_no = current_max + 1 if current_max >= 0 else 1
    batch_run_id = f"{run_id_prefix}{next_no:03d}"
    print(f"[BATCH] assigning run_id='{batch_run_id}' to {len(new)} new rows")

    # Combine with existing if append=True
    if append and not existing.empty:
        combined = pd.concat([existing, new], ignore_index=True)
        # Only fill blanks (the rows we just appended had run_id="")
        blanks = (combined["run_id"].isna()) | (combined["run_id"] == "")
        combined.loc[blanks, "run_id"] = batch_run_id
    else:
        combined = new.copy()
        combined["run_id"] = batch_run_id

    # Drop duplicates by the full identity key (keep first)
    combined = combined.drop_duplicates(
        subset=["catchment_id", "scenario", "model", "run_id"], keep="first"
    ).sort_values(["catchment_id","scenario","model","run_id"])

    combined.to_parquet(jobs_path, index=False)
    print(f"[OK] jobs written: {jobs_path}  (rows={len(combined)})")
    return combined


# Issyk-Kul example usage:
build_or_update_catchment_registry(
    catchments_root=ik_path,
    bulk_runner_root="/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda",
    # extra_columns=["notes", "region"]  # optional: preserve manual fields from existing registry
)

jobs = build_jobs_from_catchments(
    catchments_root=ik_path,
    bulk_runner_root="/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda",
    scenario="calib",
    model="CHELSA",
    global_params_path=None,  # or specify a global parameters.yml path if needed
    append=False  # overwrite jobs.parquet
)

## Build jobs manually
# jobs = pd.DataFrame([
#     # Calibration
#     dict(run_id="run_001",
#          catchment_id="kyzylsuu", parent_id="kyzylsuu",
#          scenario="calib", model="ERA5L", run_type="calib", period="1981-2010",
#          forcing_path="/abs/path/kyzylsuu_era5l.csv",
#          params_path="/abs/path/params_calib.yml",
#          settings_path="/abs/path/settings_calib.yml",
#          glacier_profile_path="/abs/path/glacier_profile.csv",
#          out_dir="bulk_runner/outputs"),
#     # Projection SSP2 with GFDL
#     dict(run_id="run_002",
#          catchment_id="kyzylsuu", parent_id="kyzylsuu",
#          scenario="SSP2", model="GFDL", run_type="projection", period="2015-2100",
#          forcing_path="/abs/path/kyzylsuu_ssp2_gfdl.csv",
#          params_path="/abs/path/params_ssp2.yml",
#          settings_path="/abs/path/settings_ssp2.yml",
#          glacier_profile_path="/abs/path/glacier_profile.csv",
#          out_dir="bulk_runner/outputs"),
# ])
#
# # Sanity checks und execution
# assert jobs["run_id"].is_unique, "run_id must be unique"
# for col in ["forcing_path","params_path"]:
#     assert jobs[col].map(lambda p: Path(p).exists()).all(), f"missing files in {col}"
#
# jobs.to_parquet("bulk_runner/data/jobs.parquet", index=False)
