from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

"""
src/ccir/paths.py

Purpose
- Centralize all filesystem locations used by the pipeline.
- Enforce `data/processed/runs/<run_id>/` as the root directory for run-scoped outputs.
- Allow every script to accept a `paths: Paths` object rather than hardcoding strings.

Design notes
- This module should be "import-safe": it performs no filesystem writes on import.
- Directory creation is explicit via ensure_*() methods.
"""

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Paths:
    """
    Canonical project paths.

    Conventions (per your outline):
    - Shared artifacts live under `data/processed/...` (claims, evidence, verdict mappings, prompts).
    - Run-scoped artifacts live under `data/processed/runs/<run_id>/...` (cache, model outputs, results, metadata, reports).
    """

    run_id: str
    repo_root: Path | None = None  # if None, inferred from this file location

    # -------------------------
    # Root helpers
    # -------------------------

    def __post_init__(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ValueError("run_id must be a non-empty string")
        if not _SAFE_RUN_ID_RE.match(self.run_id):
            raise ValueError(
                f"run_id '{self.run_id}' contains unsafe characters. "
                "Use only letters, numbers, '.', '_', '-'."
            )

        # Infer repo_root if not provided.
        # Expected layout: <repo>/src/ccir/paths.py -> parents[2] == <repo>
        if self.repo_root is None:
            inferred = Path(__file__).resolve().parents[2]
            object.__setattr__(self, "repo_root", inferred)
        else:
            object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())

        if not (self.repo_root / "src").exists() or not (self.repo_root / "data").exists():
            raise RuntimeError(f"Invalid repo root detected: {self.repo_root}")

    @property
    def data_root(self) -> Path:
        return self.repo_root / "data"  # type: ignore[operator]

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    # -------------------------
    # Shared (non-run) outputs
    # -------------------------

    @property
    def claims_dir(self) -> Path:
        return self.processed_root / "claims"

    @property
    def evidence_dir(self) -> Path:
        return self.processed_root / "evidence"

    @property
    def verdicts_dir(self) -> Path:
        return self.processed_root / "verdicts"

    @property
    def llm_prompts_dir(self) -> Path:
        return self.processed_root / "LLMprompts"

    # Claims files (shared / legacy)
    @property
    def claims_all_jsonl(self) -> Path:
        return self.claims_dir / "all.jsonl"

    @property
    def claims_for_llms_jsonl(self) -> Path:
        return self.claims_dir / "forLLMs.jsonl"

    @property
    def claims_for_scoring_jsonl(self) -> Path:
        return self.claims_dir / "forScoring.jsonl"

    # Evidence files (shared / legacy)
    @property
    def evidence_urls_jsonl(self) -> Path:
        return self.evidence_dir / "URLs.jsonl"

    @property
    def evidence_rankings_dir(self) -> Path:
        return self.evidence_dir / "rankings"

    @property
    def evidence_topk_urls_jsonl(self) -> Path:
        return self.evidence_rankings_dir / "topKURLs.jsonl"

    # Verdict mappings (shared / legacy)
    @property
    def verdicts_mapping_jsonl(self) -> Path:
        return self.verdicts_dir / "mapping.jsonl"

    # Prompts (shared / legacy)
    @property
    def small_llm_prompts_jsonl(self) -> Path:
        return self.llm_prompts_dir / "SmallLLMPrompts.jsonl"

    @property
    def judge_llm_prompts_jsonl(self) -> Path:
        return self.llm_prompts_dir / "JudgeLLMPrompts.jsonl"

    # -------------------------
    # Run-scoped outputs
    # -------------------------

    @property
    def runs_dir(self) -> Path:
        return self.processed_root / "runs"

    @property
    def run_root(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def run_metadata_jsonl(self) -> Path:
        return self.run_root / "run_metadata.jsonl"

    # -------------------------
    # Run-scoped dataset outputs (so runs don't overwrite each other)
    # -------------------------

    @property
    def run_claims_dir(self) -> Path:
        return self.run_root / "claims"

    @property
    def run_verdicts_dir(self) -> Path:
        return self.run_root / "verdicts"

    @property
    def run_claims_all_jsonl(self) -> Path:
        return self.run_claims_dir / "all.jsonl"

    @property
    def run_claims_for_llms_jsonl(self) -> Path:
        return self.run_claims_dir / "forLLMs.jsonl"

    @property
    def run_claims_for_scoring_jsonl(self) -> Path:
        return self.run_claims_dir / "forScoring.jsonl"

    @property
    def run_verdicts_mapping_jsonl(self) -> Path:
        return self.run_verdicts_dir / "mapping.jsonl"

    # -------------------------
    # Run-scoped evidence outputs (NEW)
    # -------------------------

    @property
    def run_evidence_dir(self) -> Path:
        return self.run_root / "evidence"

    @property
    def run_evidence_urls_jsonl(self) -> Path:
        return self.run_evidence_dir / "URLs.jsonl"

    @property
    def run_evidence_rankings_dir(self) -> Path:
        return self.run_evidence_dir / "rankings"

    @property
    def run_evidence_topk_urls_jsonl(self) -> Path:
        return self.run_evidence_rankings_dir / "topKURLs.jsonl"

    # Reports (run-scoped)
    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    def report_jsonl(self, step_num: int) -> Path:
        """
        Per-step report file stored under the run directory.
        Layout: runs/<run_id>/reports/report_00.jsonl
        """
        if step_num < 0 or step_num > 99:
            raise ValueError("step_num must be in [0, 99]")
        return self.reports_dir / f"report_{step_num:02d}.jsonl"

    # Cache
    @property
    def cache_dir(self) -> Path:
        return self.run_root / "cache"

    @property
    def cache_plaintext_dir(self) -> Path:
        return self.cache_dir / "plaintext"

    @property
    def cache_gold_dir(self) -> Path:
        return self.cache_dir / "gold"

    @property
    def cache_gold_docs_dir(self) -> Path:
        return self.cache_gold_dir / "gold_docs"

    @property
    def cache_corrupted_dir(self) -> Path:
        return self.cache_dir / "corrupted"

    @property
    def run_llm_prompts_dir(self) -> Path:
        return self.run_root / "LLMprompts"

    @property
    def run_small_llm_prompts_jsonl(self) -> Path:
        return self.run_llm_prompts_dir / "SmallLLMPrompts.jsonl"

    @property
    def run_judge_llm_prompts_jsonl(self) -> Path:
        return self.run_llm_prompts_dir / "JudgeLLMPrompts.jsonl"

    # Run outputs: model and judge responses
    @property
    def small_llm_responses_dir(self) -> Path:
        return self.run_root / "smallLLMResponses"

    def small_llm_responses_jsonl(self, model_name: str) -> Path:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        return self.small_llm_responses_dir / f"SmallLLMResponses{model_name}.jsonl"

    @property
    def judge_responses_dir(self) -> Path:
        return self.run_root / "LLMJudgeResponses"

    def judge_responses_jsonl(self, model_name: str) -> Path:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        return self.judge_responses_dir / f"JudgeLLMResponses{model_name}.jsonl"

    # Results
    @property
    def results_dir(self) -> Path:
        return self.run_root / "results"

    @property
    def results_model_metrics_json(self) -> Path:
        return self.results_dir / "model_metrics.json"

    @property
    def results_judge_scores_json(self) -> Path:
        return self.results_dir / "judge_scores.json"

    @property
    def results_plots_dir(self) -> Path:
        return self.results_dir / "plots"

    # -------------------------
    # Per-claim / per-url helpers
    # -------------------------

    def plaintext_path(self, claim_id: str, url_id: str) -> Path:
        """
        Cached cleaned plaintext for a URL under this run.
        Layout: runs/<run_id>/cache/plaintext/<claim_id>/<url_id>.txt
        """
        self._require_id(claim_id, "claim_id")
        self._require_id(url_id, "url_id")
        return self.cache_plaintext_dir / claim_id / f"{url_id}.txt"

    def gold_doc_path(self, claim_id: str, url_id: str) -> Path:
        """
        Gold (top-L) plaintext copy used for sentence scoring.
        Layout: runs/<run_id>/cache/gold/gold_docs/<claim_id>/<url_id>.txt
        """
        self._require_id(claim_id, "claim_id")
        self._require_id(url_id, "url_id")
        return self.cache_gold_docs_dir / claim_id / f"{url_id}.txt"

    def corrupted_doc_path(self, variant_name: str, claim_id: str, url_id: str) -> Path:
        """
        Corrupted plaintext materialization.
        Layout: runs/<run_id>/cache/corrupted/<variant_name>/<claim_id>/<url_id>.txt
        """
        if not variant_name:
            raise ValueError("variant_name must be non-empty")
        self._require_id(claim_id, "claim_id")
        self._require_id(url_id, "url_id")
        return self.cache_corrupted_dir / variant_name / claim_id / f"{url_id}.txt"

    # -------------------------
    # Directory creation (explicit)
    # -------------------------

    def ensure_shared_dirs(self) -> None:
        """Create shared directories that hold non-run artifacts. Safe to call repeatedly."""
        for d in [
            self.claims_dir,
            self.evidence_dir,
            self.evidence_rankings_dir,
            self.verdicts_dir,
            self.llm_prompts_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def ensure_run_dirs(self) -> None:
        """Create the run root and common run-scoped directories. Safe to call repeatedly."""
        for d in [
            self.run_root,
            self.reports_dir,
            self.run_llm_prompts_dir,
            self.cache_dir,
            self.cache_plaintext_dir,
            self.cache_gold_docs_dir,
            self.cache_corrupted_dir,
            self.small_llm_responses_dir,
            self.judge_responses_dir,
            self.results_dir,
            self.results_plots_dir,
            self.run_claims_dir,
            self.run_verdicts_dir,
            # NEW run-scoped evidence dirs
            self.run_evidence_dir,
            self.run_evidence_rankings_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Compatibility aliases (DEFAULT TO RUN-SCOPED)
    # -------------------------
    # Goal: existing scripts that reference paths.claims_all / paths.evidence_urls
    # automatically use run-scoped artifacts under runs/<run_id>/...
    #
    # Shared artifacts still remain available via the explicit *_jsonl properties
    # (claims_all_jsonl, evidence_urls_jsonl, etc.) for backward compatibility.

    @property
    def claims_all(self) -> Path:
        return self.run_claims_all_jsonl

    @property
    def claims_for_llms(self) -> Path:
        return self.run_claims_for_llms_jsonl

    @property
    def claims_for_scoring(self) -> Path:
        return self.run_claims_for_scoring_jsonl

    @property
    def evidence_urls(self) -> Path:
        return self.run_evidence_urls_jsonl

    @property
    def evidence_topk_urls(self) -> Path:
        return self.run_evidence_topk_urls_jsonl

    @property
    def verdicts_mapping(self) -> Path:
        return self.run_verdicts_mapping_jsonl

    # Prompts are typically shared (same prompt files reused across runs).
    # If you later decide prompts should be run-scoped, add run_* equivalents.
    @property
    def small_llm_prompts(self) -> Path:
        return self.run_small_llm_prompts_jsonl

    @property
    def judge_llm_prompts(self) -> Path:
        return self.run_judge_llm_prompts_jsonl

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _require_id(value: str, name: str) -> None:
        if not value or not isinstance(value, str):
            raise ValueError(f"{name} must be a non-empty string")
        if any(ch in value for ch in ["/", "\\"]):
            raise ValueError(f"{name} must not contain path separators: {value!r}")