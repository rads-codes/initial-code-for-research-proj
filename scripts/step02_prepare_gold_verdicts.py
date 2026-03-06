from __future__ import annotations
"""
scripts/step02_prepare_gold_verdicts.py

Step 02 (orchestrator-safe)

Inputs:
  - data/processed/claims/all.jsonl
  - data/processed/claims/forLLMs.jsonl

Output:
  - data/processed/runs/<run_id>/claims/forScoring.jsonl
"""

import argparse
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ccir.io_utils import read_jsonl, write_jsonl_atomic, append_jsonl
from ccir.paths import Paths
from ccir.schemas import ScoringClaimRow, utc_now_iso


def _code_version() -> str:
    return os.getenv("CCIR_CODE_VERSION", "dev")


def build_for_scoring_rows(*, paths: Paths) -> List[ScoringClaimRow]:
    all_rows = read_jsonl(paths.claims_all)
    llm_rows = read_jsonl(paths.claims_for_llms)

    ratings_by_id: Dict[str, str] = {}
    for r in all_rows:
        claim_id = r.get("claim_id")
        if not claim_id:
            raise ValueError(f"claims/all.jsonl row missing claim_id: {r}")

        if claim_id in ratings_by_id:
            raise ValueError(f"Duplicate claim_id in claims/all.jsonl: {claim_id}")

        rating = r.get("rating")
        if rating is None or (isinstance(rating, str) and rating.strip() == ""):
            raise ValueError(f"Missing/empty rating for claim_id={claim_id} in claims/all.jsonl")

        ratings_by_id[str(claim_id)] = str(rating)

    out: List[ScoringClaimRow] = []
    for r in llm_rows:
        claim_id = r.get("claim_id")
        if not claim_id:
            raise ValueError(f"claims/forLLMs.jsonl row missing claim_id: {r}")

        cid = str(claim_id)
        if cid not in ratings_by_id:
            raise ValueError(f"claim_id={cid} appears in claims/forLLMs.jsonl but not in claims/all.jsonl")

        out.append(
            ScoringClaimRow(
                run_id=paths.run_id,
                created_utc=utc_now_iso(),
                code_version=_code_version(),
                claim_id=cid,
                rating=ratings_by_id[cid],
            )
        )

    return out


def run_step02(
    *,
    paths: Paths,
    config: Any = None,          # orchestrator may pass; unused here
    logger: Any = None,          # orchestrator logger (step_logger ctx)
    step_logger: Any = None,     # alias
    log: Any = None,             # alias
    overwrite: bool = True,      # safe default: regenerate forScoring each run
    **_: Any,
) -> None:
    """
    Orchestrator entrypoint. DO NOT parse argv.
    """
    # pick whichever logger alias exists
    _logger = logger or step_logger or log

    rows = build_for_scoring_rows(paths=paths)
    write_jsonl_atomic(paths.claims_for_scoring, rows=[asdict(r) for r in rows])

    # write a lightweight report row too (optional, but handy)
    report_row = {
        "step": "02",
        "created_utc": utc_now_iso(),
        "run_id": paths.run_id,
        "code_version": _code_version(),
        "counts": {
            "claims_in_forLLMs": len(read_jsonl(paths.claims_for_llms)),
            "claims_written_forScoring": len(rows),
        },
        "inputs": [str(paths.claims_all), str(paths.claims_for_llms)],
        "outputs": [str(paths.claims_for_scoring)],
    }
    append_jsonl(paths.report_jsonl(2), report_row)

    # best-effort counters into orchestrator report
    if _logger is not None:
        fn = getattr(_logger, "count", None)
        if callable(fn):
            fn("claims_written_forScoring", len(rows))


def main() -> int:
    """
    Standalone CLI (optional). This is NOT used by the orchestrator.
    """
    ap = argparse.ArgumentParser(description="Step 02: prepare gold verdicts for scoring.")
    ap.add_argument("--run-id", required=True, help="Run identifier used for run-scoped outputs.")
    ap.add_argument("--repo-root", default=None, help="Optional repo root override.")
    args = ap.parse_args()

    paths = Paths(run_id=args.run_id, repo_root=args.repo_root)
    if hasattr(paths, "ensure_run_dirs"):
        paths.ensure_run_dirs()

    # reuse orchestrator entrypoint
    run_step02(paths=paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())