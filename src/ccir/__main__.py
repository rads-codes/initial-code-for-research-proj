from __future__ import annotations
import argparse
import dataclasses
import importlib
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

"""
src/ccir/__main__.py

Pipeline orchestrator.

Doc requirements:
- call load_config(), load_env_file_if_present(), build Paths(run_id)
- wrap each step with logging_utils + run_metadata
- call validation at start/end of each step and before compute-heavy steps

This file is import-safe; orchestration only runs under __main__.
"""

from ccir.config_loader import load_config
from ccir.paths import Paths
from ccir.logging_utils import step_logger
from ccir.utils.env_loader import load_env_file_if_present
from ccir.utils.hashing import sha256_hex_text
from ccir.utils.run_metadata import RunMetadataLogger
from ccir.validation import validate_step


# -----------------------------
# Step registry
# -----------------------------

# NOTE:
# - "entrypoint" indicates which function __main__ should call in the step module.
# - If omitted, __main__ falls back to run(...), else main(...).
STEPS: List[Dict[str, Any]] = [
    {"id": "00", "name": "step00_prepare_dataset", "module": "scripts.step00_prepare_dataset", "entrypoint": "run_step00"},
    {"id": "01", "name": "step01_small_LLMs_dataset", "module": "scripts.step01_small_LLMs_dataset", "entrypoint": "run_step01"},
    {"id": "02", "name": "step02_prepare_gold_verdicts", "module": "scripts.step02_prepare_gold_verdicts", "entrypoint": "run_step02"},
    {"id": "03", "name": "step03_collect_URLS", "module": "scripts.step03_collect_URLS", "entrypoint": "run_step03"},
    {"id": "04", "name": "step04_cache_URL_content", "module": "scripts.step04_cache_URL_content", "entrypoint": "run_step04"},
    {"id": "05", "name": "step05_BM25_ranking", "module": "scripts.step05_BM25_ranking", "entrypoint": "run_step05"},
    {"id": "06", "name": "step06_build_gold", "module": "scripts.step06_build_gold", "entrypoint": "run_step06"},
    {"id": "07", "name": "step07_sentences_cosine_similarity", "module": "scripts.step07_sentences_cosine_similarity", "entrypoint": "run_step07"},
    {"id": "08", "name": "step08_make_corruptions", "module": "scripts.step08_make_corruptions", "entrypoint": "run_step08"},
    {"id": "09", "name": "step09_run_models", "module": "scripts.step09_run_models", "entrypoint": "run_step09"},
    {"id": "10", "name": "step10_run_judge", "module": "scripts.step10_run_judge", "entrypoint": "run_step10"},
    {"id": "11", "name": "step11_results", "module": "scripts.step11_results", "entrypoint": "run_step11"},
]

# “Before compute-heavy steps”
HEAVY_STEP_IDS = {"04", "07", "08", "09", "10"}


# -----------------------------
# Helpers
# -----------------------------

def _stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization for hashing."""
    if dataclasses.is_dataclass(obj):
        payload = dataclasses.asdict(obj)
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        payload = obj.to_dict()
    elif hasattr(obj, "__dict__"):
        payload = obj.__dict__
    else:
        payload = obj
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _compute_config_hash(config: Any) -> str:
    return sha256_hex_text(_stable_json_dumps(config))


def _pick_steps(
    *,
    step_ids_csv: Optional[str],
    start_id: Optional[str],
    end_id: Optional[str],
) -> List[Dict[str, Any]]:
    by_id = {s["id"]: s for s in STEPS}

    if step_ids_csv:
        wanted: List[Dict[str, Any]] = []
        for raw in step_ids_csv.split(","):
            sid = raw.strip()
            if not sid:
                continue
            if sid not in by_id:
                raise ValueError(f"Unknown step id {sid!r}. Known: {sorted(by_id.keys())}")
            wanted.append(by_id[sid])
        return wanted

    if start_id is None and end_id is None:
        return list(STEPS)

    ids = [s["id"] for s in STEPS]
    if start_id is None:
        start_id = ids[0]
    if end_id is None:
        end_id = ids[-1]
    if start_id not in by_id or end_id not in by_id:
        raise ValueError(f"Invalid range: start={start_id!r} end={end_id!r}")

    start_idx = ids.index(start_id)
    end_idx = ids.index(end_id)
    if end_idx < start_idx:
        raise ValueError(f"Invalid range: end ({end_id}) is before start ({start_id})")

    return STEPS[start_idx : end_idx + 1]


def _call_entrypoint_best_effort(mod: Any, *, entrypoint: Optional[str], kwargs: Dict[str, Any]) -> None:
    """
    Preferred call order:
      1) explicit entrypoint if provided (e.g. run_step01)
      2) run(...)
      3) main(...)

    Passes only supported kwargs based on function signature.
    """
    fn: Optional[Callable[..., Any]] = None

    if entrypoint and hasattr(mod, entrypoint) and callable(getattr(mod, entrypoint)):
        fn = getattr(mod, entrypoint)
    elif hasattr(mod, "run") and callable(getattr(mod, "run")):
        fn = getattr(mod, "run")
    elif hasattr(mod, "main") and callable(getattr(mod, "main")):
        fn = getattr(mod, "main")

    if fn is None:
        raise AttributeError(
            f"Step module {mod.__name__} must define {entrypoint}(...) or run(...) or main(...)."
        )

    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    fn(**filtered)  # type: ignore[misc]


def _validate_safe(*, stage: str, step_id: str, paths: Paths, config: Any) -> None:
    """Handle validate_step signatures that may accept different stage names."""
    try:
        validate_step(stage=stage, step=step_id, paths=paths, config=config)
    except TypeError:
        if stage in ("pre_heavy", "heavy_pre"):
            validate_step(stage="pre", step=step_id, paths=paths, config=config)
        elif stage in ("post_heavy", "heavy_post"):
            validate_step(stage="post", step=step_id, paths=paths, config=config)
        else:
            raise


def _required_keys_for_selected_steps(step_ids: Sequence[str]) -> List[str]:
    keys: List[str] = []
    if any(sid in step_ids for sid in ("03",)):
        keys.append("SERPAPI_API_KEY")
    if any(sid in step_ids for sid in ("09", "10")):
        keys.append("OPENROUTER_API_KEY")
    out: List[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ccir", description="Run the CCIR pipeline.")
    p.add_argument("run_id", help="Run id, used for data/processed/runs/<run_id>/...")
    p.add_argument("--steps", default=None, help="Comma-separated step ids, e.g. 00,01,02")
    p.add_argument("--start", default=None, help="Start step id (inclusive), e.g. 03")
    p.add_argument("--end", default=None, help="End step id (inclusive), e.g. 08")
    p.add_argument("--dry-run", action="store_true", help="Print what would run, then exit.")

    p.add_argument("--env-file", default=None, help="Path to api_keys.env (optional).")
    p.add_argument("--override-env", action="store_true", help="Allow env file to override shell env.")
    p.add_argument(
        "--require-keys",
        default=None,
        help="Comma-separated env keys to require (overrides auto-inference).",
    )

    p.add_argument("--code-version", default=None, help="Manual code version string for lineage.")
    return p


# -----------------------------
# Main orchestration
# -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    selected = _pick_steps(step_ids_csv=args.steps, start_id=args.start, end_id=args.end)
    selected_ids = [s["id"] for s in selected]

    # Env load early
    if args.require_keys is not None:
        required_keys = [k.strip() for k in args.require_keys.split(",") if k.strip()]
    else:
        required_keys = _required_keys_for_selected_steps(selected_ids)

    load_env_file_if_present(
        env_path=args.env_file,
        override=bool(args.override_env),
        required_keys=required_keys if required_keys else None,
    )

    # Config
    config = load_config()
    if hasattr(config, "validate") and callable(getattr(config, "validate")):
        config.validate()

    code_version = (
        args.code_version
        or getattr(config, "code_version", None)
        or os.getenv("CCIR_CODE_VERSION")
        or "dev"
    )

    # Paths
    paths = Paths(run_id=args.run_id)
    if hasattr(paths, "ensure_shared_dirs"):
        paths.ensure_shared_dirs()
    if hasattr(paths, "ensure_run_dirs"):
        paths.ensure_run_dirs()

    # Config hash (stable)
    config_hash = _compute_config_hash(config)

    if args.dry_run:
        print(f"run_id={args.run_id} code_version={code_version} config_hash={config_hash[:12]}...")
        print("steps:")
        for s in selected:
            ep = s.get("entrypoint")
            print(f"  - {s['id']}: {s['name']} ({s['module']}:{ep or 'run/main'})")
        return 0

    # Run metadata
    rm = RunMetadataLogger(paths=paths, code_version=code_version, config_hash=config_hash)
    if hasattr(rm, "log_run_start"):
        rm.log_run_start()

    try:
        for s in selected:
            step_id: str = s["id"]
            step_name: str = s["name"]
            module_path: str = s["module"]
            entrypoint: Optional[str] = s.get("entrypoint")

            # Pre-validation
            _validate_safe(stage="pre", step_id=step_id, paths=paths, config=config)
            if step_id in HEAVY_STEP_IDS:
                _validate_safe(stage="pre_heavy", step_id=step_id, paths=paths, config=config)

            step_num = int(step_id)
            report_path = paths.report_jsonl(step_num) if hasattr(paths, "report_jsonl") else None
            if report_path is None:
                raise RuntimeError("Paths.report_jsonl(step_num) is required for per-step reports.")

            with rm.step(step=step_name):
                with step_logger(
                    report_path,
                    run_id=paths.run_id,
                    code_version=code_version,
                    step=step_name,
                    start_message="step_start",
                    start_fields={"step_id": step_id, "module": module_path, "entrypoint": entrypoint},
                ) as log:
                    mod = importlib.import_module(module_path)

                    # Pass common aliases; steps can accept whichever they want.
                    _call_entrypoint_best_effort(
                        mod,
                        entrypoint=entrypoint,
                        kwargs={
                            "paths": paths,
                            "config": config,
                            "run_id": paths.run_id,
                            "code_version": code_version,
                            "config_hash": config_hash,
                            "log": log,
                            "logger": log,
                            "step_logger": log,
                        },
                    )

            # Post-validation
            _validate_safe(stage="post", step_id=step_id, paths=paths, config=config)

    except Exception:
        try:
            if hasattr(rm, "log_run_end"):
                rm.log_run_end()
        finally:
            raise

    if hasattr(rm, "log_run_end"):
        rm.log_run_end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
How to run (repo root, with src/ as sources root):

python -m ccir pilot1 --steps 01
python -m ccir pilot1 --steps 00,01,02
python -m ccir pilot1 --start 03 --end 06
python -m ccir pilot1 --start 03 --end 06 --dry-run
python -m ccir pilot1 --env-file api_keys.env --override-env
"""