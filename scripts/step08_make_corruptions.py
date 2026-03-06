from __future__ import annotations

'''
step08_make_corruptions.py
Input: data/processed/runs/<run_id>/cache/gold/<claim_id>/<URL_ID>/embeddings.jsonl, src/ccir/corruption/strategies, and src/ccir/corruption/materialize
Process: Determine which of these corruption settings are actually used and what the actual percentages are from configs.py. This then makes multiple corrupted versions of the gold evidence: random drop (decides on pseudorandom 20%, 40%, 60% of sentences to delete using sentence IDs), targeted drop (top 20%, top 40%, top 60% sentences using sentence IDs with scores in top ##%), replacement mix (random 20% replaced with random sentences, random 40% replaced with random sentences, random 60% replaced with random sentences) using src/ccir/corruption/strategies and src/ccir/corruption/materialize. It reads data/processed/runs/<run_id>/cache/gold/claimID/article/embeddings.jsonl and random_paragraphs.txt. It writes variant datastores.
Outputs: writes all plaintext files under data/processed/runs/<run_id>/cache/corrupted/variant/<claim_id>/<URL_ID>
'''

"""
scripts/08_make_corruptions.py

Input:
  - data/processed/runs/<run_id>/cache/gold/<claim_id>/<url_id>/embeddings.jsonl
  - data/raw/random_paragraphs.txt
  - optionally: ccir.corruption.strategies
  - optionally: ccir.corruption.materialize

Process:
  - Read sentence rows + similarity scores from embeddings.jsonl
  - Use config.corruption.methods / levels / random_seed
  - Build corrupted variants:
      * random_drop
      * targeted_drop
      * replacement_mix
  - Write plaintext files under:
      data/processed/runs/<run_id>/cache/corrupted/<variant_name>/<claim_id>/<url_id>.txt

Output:
  - One corrupted plaintext file per enabled (method, level) pair
"""

import argparse
import importlib
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from ccir.config_loader import load_config
from ccir.io_utils import ensure_parent_dir, read_jsonl, read_text, write_text_atomic
from ccir.paths import Paths


# ---------------------------------------------------------------------
# Logger helpers
# ---------------------------------------------------------------------
def _log(logger: Any, level: str, message: str, **kwargs: Any) -> None:
    if logger is None:
        extras = f" | {kwargs}" if kwargs else ""
        print(f"[{level.upper()}] {message}{extras}")
        return

    fn = getattr(logger, level, None)
    if callable(fn):
        try:
            fn(message, **kwargs)
            return
        except TypeError:
            try:
                fn(message)
                return
            except Exception:
                pass

    generic = getattr(logger, "log", None)
    if callable(generic):
        try:
            generic(level=level, message=message, **kwargs)
            return
        except Exception:
            pass

    extras = f" | {kwargs}" if kwargs else ""
    print(f"[{level.upper()}] {message}{extras}")


# ---------------------------------------------------------------------
# Sentence parsing helpers
# ---------------------------------------------------------------------
_SENT_NUM_RE = re.compile(r"(\d+)")


def _extract_sentence_id(row: Dict[str, Any]) -> str:
    for key in ("sentence_id", "sent_id", "id", "sentenceID"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_sentence_text(row: Dict[str, Any]) -> str:
    for key in ("sentence_text", "sentence", "text", "sentenceText"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_score(row: Dict[str, Any]) -> float:
    for key in ("embedding_score", "score", "cosine_similarity", "similarity"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            continue
    return 0.0


def _sentence_order(sentence_id: str) -> int:
    if not sentence_id:
        return 10**9
    match = _SENT_NUM_RE.search(sentence_id)
    if not match:
        return 10**9
    try:
        return int(match.group(1))
    except Exception:
        return 10**9


def _load_sentences_from_embeddings(embeddings_path: Path) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(embeddings_path))
    out: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        sentence_text = _extract_sentence_text(row)
        if not sentence_text:
            continue

        out.append(
            {
                "sentence_id": _extract_sentence_id(row),
                "sentence_text": sentence_text,
                "embedding_score": _extract_score(row),
            }
        )

    out.sort(key=lambda r: _sentence_order(r["sentence_id"]))
    return out


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------
def _normalize_method(method: Any) -> str:
    if hasattr(method, "value"):
        method = getattr(method, "value")
    return str(method).strip()


def _get_corruption_methods(config: Any) -> List[str]:
    methods = getattr(config.corruption, "methods", [])
    out: List[str] = []
    for m in methods:
        name = _normalize_method(m)
        if name:
            out.append(name)
    return out


def _get_corruption_levels_pct(config: Any) -> List[int]:
    levels = getattr(config.corruption, "levels", [])
    out: List[int] = []
    for lvl in levels:
        pct = int(round(float(lvl) * 100))
        if pct > 0:
            out.append(pct)
    return out


def _get_corruption_seed(config: Any) -> int:
    try:
        return int(getattr(config.corruption, "random_seed", 12345))
    except Exception:
        return 12345


# ---------------------------------------------------------------------
# Replacement pool
# ---------------------------------------------------------------------
def _load_replacement_pool(paths: Paths) -> List[str]:
    candidates = [
        paths.raw_root / "random_paragraphs.txt",
        paths.repo_root / "data" / "raw" / "random_paragraphs.txt",
        Path("data/raw/random_paragraphs.txt"),
    ]

    for path in candidates:
        if path.exists():
            text = read_text(path)
            return [line.strip() for line in text.splitlines() if line.strip()]

    return []


# ---------------------------------------------------------------------
# Corruption strategy fallbacks
# ---------------------------------------------------------------------
def _num_to_modify(n_sentences: int, pct: int) -> int:
    if n_sentences <= 0:
        return 0
    k = int(round((pct / 100.0) * n_sentences))
    return max(1, min(n_sentences, k))


def _random_drop_indices(n_sentences: int, pct: int, rng: random.Random) -> List[int]:
    k = _num_to_modify(n_sentences, pct)
    return sorted(rng.sample(range(n_sentences), k))


def _targeted_drop_indices(sentences: Sequence[Dict[str, Any]], pct: int) -> List[int]:
    k = _num_to_modify(len(sentences), pct)
    ranked = sorted(
        enumerate(sentences),
        key=lambda pair: (-float(pair[1].get("embedding_score", 0.0)), pair[0]),
    )
    return sorted(idx for idx, _ in ranked[:k])


def _replacement_indices(n_sentences: int, pct: int, rng: random.Random) -> List[int]:
    k = _num_to_modify(n_sentences, pct)
    return sorted(rng.sample(range(n_sentences), k))


# ---------------------------------------------------------------------
# Materialization fallbacks
# ---------------------------------------------------------------------
def _materialize_drop(sentences: Sequence[Dict[str, Any]], drop_indices: Sequence[int]) -> str:
    drop_set = set(drop_indices)
    kept = [
        row["sentence_text"]
        for i, row in enumerate(sentences)
        if i not in drop_set and row.get("sentence_text")
    ]
    return "\n".join(kept).strip() + "\n"


def _materialize_replace(
    sentences: Sequence[Dict[str, Any]],
    replace_indices: Sequence[int],
    replacement_pool: Sequence[str],
    rng: random.Random,
) -> str:
    replace_set = set(replace_indices)

    if not replacement_pool:
        return _materialize_drop(sentences, replace_indices)

    out: List[str] = []
    for i, row in enumerate(sentences):
        if i in replace_set:
            out.append(rng.choice(replacement_pool).strip())
        else:
            out.append(row["sentence_text"])

    return "\n".join(x for x in out if x.strip()).strip() + "\n"


# ---------------------------------------------------------------------
# Optional delegation into ccir.corruption modules
# ---------------------------------------------------------------------
def _maybe_import(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _try_strategy_module(
    strategies_mod: Any,
    *,
    method: str,
    sentences: Sequence[Dict[str, Any]],
    pct: int,
    rng: random.Random,
) -> Optional[List[int]]:
    candidates = {
        "random_drop": ["pick_random_drop", "random_drop", "choose_random_drop"],
        "targeted_drop": ["pick_targeted_drop", "targeted_drop", "choose_targeted_drop"],
        "replacement_mix": ["pick_replacements", "replacement_mix", "choose_replacements"],
    }.get(method, [])

    for name in candidates:
        fn = getattr(strategies_mod, name, None)
        if not callable(fn):
            continue

        try:
            result = fn(sentences=sentences, pct=pct, rng=rng)
            if isinstance(result, list):
                return result
        except TypeError:
            pass
        except Exception:
            pass

        try:
            result = fn(sentences, pct, rng)
            if isinstance(result, list):
                return result
        except Exception:
            pass

    return None


def _try_materialize_module(
    materialize_mod: Any,
    *,
    method: str,
    sentences: Sequence[Dict[str, Any]],
    chosen_indices: Sequence[int],
    replacement_pool: Sequence[str],
    rng: random.Random,
) -> Optional[str]:
    candidates = {
        "random_drop": ["materialize_drop", "apply_drop", "render_drop"],
        "targeted_drop": ["materialize_drop", "apply_drop", "render_drop"],
        "replacement_mix": ["materialize_replace", "apply_replace", "render_replace"],
    }.get(method, [])

    for name in candidates:
        fn = getattr(materialize_mod, name, None)
        if not callable(fn):
            continue

        try:
            if method == "replacement_mix":
                result = fn(
                    sentences=sentences,
                    replace_indices=chosen_indices,
                    replacement_pool=replacement_pool,
                    rng=rng,
                )
            else:
                result = fn(sentences=sentences, drop_indices=chosen_indices)
            if isinstance(result, str):
                return result
        except TypeError:
            pass
        except Exception:
            pass

        try:
            if method == "replacement_mix":
                result = fn(sentences, chosen_indices, replacement_pool, rng)
            else:
                result = fn(sentences, chosen_indices)
            if isinstance(result, str):
                return result
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------
def _find_embeddings_files(paths: Paths) -> List[Path]:
    gold_dir = paths.cache_gold_dir
    if not gold_dir.exists():
        return []

    found: List[Path] = []
    for path in gold_dir.glob("*/*/embeddings.jsonl"):
        parts = path.relative_to(gold_dir).parts
        if len(parts) == 3 and parts[0] != "gold_docs":
            found.append(path)
    return sorted(found)


def _infer_claim_id_and_url_id(paths: Paths, embeddings_path: Path) -> Tuple[str, str]:
    rel = embeddings_path.relative_to(paths.cache_gold_dir)
    claim_id = rel.parts[0]
    url_id = rel.parts[1]
    return claim_id, url_id


def _variant_name(method: str, pct: int) -> str:
    return f"{method}_{pct}"


# ---------------------------------------------------------------------
# Main step logic
# ---------------------------------------------------------------------
def run_step08(
    *,
    paths: Paths,
    config: Any,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    config_hash: Optional[str] = None,
    log: Any = None,
    logger: Any = None,
    step_logger: Any = None,
) -> Dict[str, int]:
    active_logger = logger or log or step_logger

    methods = _get_corruption_methods(config)
    levels_pct = _get_corruption_levels_pct(config)
    base_seed = _get_corruption_seed(config)

    replacement_pool = _load_replacement_pool(paths)
    strategies_mod = _maybe_import("ccir.corruption.strategies")
    materialize_mod = _maybe_import("ccir.corruption.materialize")

    embeddings_files = _find_embeddings_files(paths)

    counts: Dict[str, int] = {
        "embeddings_files_found": 0,
        "articles_loaded": 0,
        "articles_skipped_empty": 0,
        "articles_skipped_bad_input": 0,
        "variants_written": 0,
        "replacement_pool_size": len(replacement_pool),
    }

    _log(
        active_logger,
        "info",
        "step08_make_corruptions starting",
        run_id=run_id or paths.run_id,
        methods=methods,
        levels_pct=levels_pct,
        random_seed=base_seed,
        replacement_pool_size=len(replacement_pool),
    )

    if not methods:
        raise ValueError("config.corruption.methods is empty")
    if not levels_pct:
        raise ValueError("config.corruption.levels is empty")
    if not embeddings_files:
        _log(active_logger, "warning", "No embeddings.jsonl files found under cache/gold")
        return counts

    counts["embeddings_files_found"] = len(embeddings_files)

    progress = tqdm(
        embeddings_files,
        desc="Step 08 corruptions",
        unit="article",
        dynamic_ncols=True,
        leave=True,
    )

    for embeddings_path in progress:
        claim_id, url_id = _infer_claim_id_and_url_id(paths, embeddings_path)

        try:
            sentences = _load_sentences_from_embeddings(embeddings_path)
        except Exception as e:
            counts["articles_skipped_bad_input"] += 1
            _log(
                active_logger,
                "warning",
                "Skipping malformed embeddings file",
                claim_id=claim_id,
                url_id=url_id,
                path=str(embeddings_path),
                error=str(e),
            )
            continue

        if not sentences:
            counts["articles_skipped_empty"] += 1
            _log(
                active_logger,
                "warning",
                "Skipping empty embeddings file",
                claim_id=claim_id,
                url_id=url_id,
                path=str(embeddings_path),
            )
            continue

        counts["articles_loaded"] += 1
        n_sentences = len(sentences)

        for method in methods:
            for pct in levels_pct:
                variant = _variant_name(method, pct)
                rng = random.Random(f"{base_seed}:{claim_id}:{url_id}:{method}:{pct}")

                chosen_indices: Optional[List[int]] = None

                if strategies_mod is not None:
                    chosen_indices = _try_strategy_module(
                        strategies_mod,
                        method=method,
                        sentences=sentences,
                        pct=pct,
                        rng=rng,
                    )

                if chosen_indices is None:
                    if method == "random_drop":
                        chosen_indices = _random_drop_indices(n_sentences, pct, rng)
                    elif method == "targeted_drop":
                        chosen_indices = _targeted_drop_indices(sentences, pct)
                    elif method == "replacement_mix":
                        chosen_indices = _replacement_indices(n_sentences, pct, rng)
                    else:
                        _log(
                            active_logger,
                            "warning",
                            "Unknown corruption method; skipping",
                            method=method,
                            claim_id=claim_id,
                            url_id=url_id,
                        )
                        continue

                corrupted_text: Optional[str] = None

                if materialize_mod is not None:
                    corrupted_text = _try_materialize_module(
                        materialize_mod,
                        method=method,
                        sentences=sentences,
                        chosen_indices=chosen_indices,
                        replacement_pool=replacement_pool,
                        rng=rng,
                    )

                if corrupted_text is None:
                    if method in ("random_drop", "targeted_drop"):
                        corrupted_text = _materialize_drop(sentences, chosen_indices)
                    elif method == "replacement_mix":
                        corrupted_text = _materialize_replace(
                            sentences,
                            chosen_indices,
                            replacement_pool,
                            rng,
                        )
                    else:
                        continue

                out_path = paths.corrupted_doc_path(variant, claim_id, url_id)
                ensure_parent_dir(out_path)
                write_text_atomic(out_path, corrupted_text)
                counts["variants_written"] += 1

        progress.set_postfix(variants=counts["variants_written"])

    progress.close()

    _log(active_logger, "info", "step08_make_corruptions finished", counts=counts)
    return counts


# ---------------------------------------------------------------------
# Direct script CLI support
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step 08: materialize corrupted evidence variants")
    p.add_argument("--run-id", required=True, help="Run id, e.g. pilot1")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config()
    paths = Paths(run_id=args.run_id)
    paths.ensure_run_dirs()
    run_step08(paths=paths, config=config, run_id=args.run_id)


if __name__ == "__main__":
    main()