from __future__ import annotations

"""
scripts/step11_results.py

Inputs
------
- runs/<run_id>/claims/forScoring.jsonl
- runs/<run_id>/claims/forLLMs.jsonl
- runs/<run_id>/smallLLMResponses/SmallLLMResponses*.jsonl
- runs/<run_id>/LLMJudgeResponses/JudgeLLMResponses*.jsonl

Outputs
-------
- runs/<run_id>/results/model_metrics.json
- runs/<run_id>/results/judge_scores.json
- runs/<run_id>/results/per_example_rows.jsonl
- runs/<run_id>/results/per_example_rows.csv
- runs/<run_id>/results/plots/*.png

Behavior
--------
- Compares small-model verdicts to gold labels
- Aggregates judge scores onto each small-model response row
- Computes metrics by model / condition / language
- Produces confusion matrices and summary plots
"""

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt

from ccir.paths import Paths

try:
    from ccir.io_utils import read_jsonl, write_jsonl_atomic, ensure_parent_dir
except Exception:
    read_jsonl = None
    write_jsonl_atomic = None
    ensure_parent_dir = None


LABELS: List[str] = [
    "Supported",
    "Refuted",
    "Not_Enough_Evidence",
    "Conflicting_Evidence",
]

JUDGE_DIMS: List[str] = [
    "political_bias",
    "sociocultural_bias",
    "linguistic_bias",
    "logic_of_reasoning",
    "evidence_usage",
]


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def _ensure_parent(path: Path) -> None:
    if ensure_parent_dir is not None:
        ensure_parent_dir(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    if read_jsonl is not None:
        rows = read_jsonl(path)
        return [dict(r) if not isinstance(r, dict) else r for r in rows]

    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_json(path: Path, obj: Any) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if write_jsonl_atomic is not None:
        write_jsonl_atomic(path, rows)
        return

    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------
def _log(logger: Any, message: str, **kwargs: Any) -> None:
    if logger is None:
        return
    payload = {"message": message, **kwargs}
    for name in ("info", "log"):
        fn = getattr(logger, name, None)
        if callable(fn):
            try:
                fn(payload)
                return
            except Exception:
                pass


def _count(logger: Any, key: str, value: int = 1) -> None:
    if logger is None:
        return
    fn = getattr(logger, "count", None)
    if callable(fn):
        try:
            fn(key, value)
        except Exception:
            pass


# ---------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------
def _safe_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "unnamed"


def _safe_plot(logger: Any, name: str, fn, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as e:
        _log(logger, "plot_failed", plot_name=name, error=str(e))


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
def _s(x: Any) -> Optional[str]:
    if x is None:
        return None
    y = str(x).strip()
    return y if y else None


def normalize_label(label: Any) -> Optional[str]:
    s = _s(label)
    if s is None:
        return None

    t = s.lower().strip().replace("-", "_").replace(" ", "_")
    mapping = {
        "supported": "Supported",
        "refuted": "Refuted",
        "not_enough_evidence": "Not_Enough_Evidence",
        "not_enough_info": "Not_Enough_Evidence",
        "insufficient_evidence": "Not_Enough_Evidence",
        "nei": "Not_Enough_Evidence",
        "conflicting_evidence": "Conflicting_Evidence",
        "conflicting": "Conflicting_Evidence",
    }
    return mapping.get(t, s if s in LABELS else None)


def _claim_id(row: Dict[str, Any]) -> Optional[str]:
    x = row.get("claim_id")
    return None if x is None else str(x)


def _model_aliases(row: Dict[str, Any]) -> List[str]:
    """
    Return all possible identifiers for the judged / small model.
    We prefer keeping both ids and names because different files use different ones.
    """
    aliases: List[str] = []
    for key in ("small_model_id", "model_id", "small_model_name", "model_name"):
        v = _s(row.get(key))
        if v and v not in aliases:
            aliases.append(v)
    return aliases


def _primary_model_key(row: Dict[str, Any]) -> str:
    """
    Human-readable canonical key for grouping plots/metrics.
    Prefer model name when available so outputs are interpretable.
    """
    return (
        _s(row.get("small_model_name"))
        or _s(row.get("model_name"))
        or _s(row.get("small_model_id"))
        or _s(row.get("model_id"))
        or "unknown_model"
    )


def _judge_key(row: Dict[str, Any]) -> str:
    return _s(row.get("judge_id")) or _s(row.get("judge_name")) or "unknown_judge"


def _condition_key(row: Dict[str, Any]) -> str:
    return _s(row.get("variant_name")) or _s(row.get("condition")) or "gold"


def _condition_bucket(row: Dict[str, Any]) -> str:
    c = _s(row.get("condition"))
    if c:
        return c
    v = _s(row.get("variant_name"))
    return "gold" if not v or v == "gold" else "corrupted"


def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_severity(variant_name: Optional[str]) -> Optional[float]:
    if not variant_name or variant_name == "gold":
        return 0.0
    digits = []
    cur = ""
    for ch in variant_name:
        if ch.isdigit() or ch == ".":
            cur += ch
        elif cur:
            digits.append(cur)
            cur = ""
    if cur:
        digits.append(cur)
    if not digits:
        return None
    try:
        v = float(digits[0])
        return v / 100.0 if v > 1 else v
    except Exception:
        return None


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return mean(xs) if xs else None


def confusion_matrix_counts(
    y_true: List[str],
    y_pred: List[str],
    labels: Sequence[str],
) -> List[List[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            mat[idx[yt]][idx[yp]] += 1
    return mat


def classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    mat = confusion_matrix_counts(y_true, y_pred, labels)
    total = sum(sum(row) for row in mat)
    correct = sum(mat[i][i] for i in range(len(labels)))
    accuracy = (correct / total) if total else None

    per_class: Dict[str, Dict[str, Any]] = {}
    macro_f1_vals: List[float] = []
    weighted_sum = 0.0
    weighted_n = 0

    for i, label in enumerate(labels):
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(len(labels)) if r != i)
        fn = sum(mat[i][c] for c in range(len(labels)) if c != i)
        support = sum(mat[i])

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1_vals.append(f1)
        weighted_sum += f1 * support
        weighted_n += support

    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": (sum(macro_f1_vals) / len(macro_f1_vals)) if macro_f1_vals else None,
        "weighted_f1": (weighted_sum / weighted_n) if weighted_n else None,
        "labels": list(labels),
        "per_class": per_class,
        "confusion_matrix": mat,
    }


def _group_by(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k) for k in keys)].append(row)
    return out


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _save_plot(fig: plt.Figure, path: Path) -> None:
    _ensure_parent(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar(
    rows: List[Dict[str, Any]],
    x_key: str,
    series_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    xs = sorted({str(r[x_key]) for r in rows if r.get(x_key) is not None})
    series = sorted({str(r[series_key]) for r in rows if r.get(series_key) is not None})
    if not xs or not series:
        return

    lookup = {(str(r[x_key]), str(r[series_key])): r.get(y_key) for r in rows}
    width = 0.8 / max(1, len(series))

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(xs)))

    for si, s in enumerate(series):
        x_positions = []
        ys = []
        for i, x in enumerate(xs):
            x_positions.append(i - 0.4 + width / 2 + si * width)
            y = lookup.get((x, s))
            ys.append(0.0 if y is None else float(y))
        ax.bar(x_positions, ys, width=width, label=s)

    ax.set_xticks(positions)
    ax.set_xticklabels(xs, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save_plot(fig, path)


def _plot_line(
    rows: List[Dict[str, Any]],
    x_key: str,
    series_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    series = sorted({str(r[series_key]) for r in rows if r.get(series_key) is not None})
    xs = sorted({float(r[x_key]) for r in rows if r.get(x_key) is not None})
    if not series or not xs:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for s in series:
        subset = [r for r in rows if str(r.get(series_key)) == s]
        lookup = {float(r[x_key]): r.get(y_key) for r in subset if r.get(x_key) is not None}
        ys = [lookup.get(x) for x in xs]
        ys = [float(y) if y is not None else math.nan for y in ys]
        ax.plot(xs, ys, marker="o", label=s)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save_plot(fig, path)


def _plot_confusion_matrix(mat: List[List[int]], labels: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", fontsize=8)

    _save_plot(fig, path)


def _plot_heatmap(
    matrix: List[List[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    path: Path,
) -> None:
    if not matrix or not row_labels or not col_labels:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.1), max(4, len(row_labels) * 0.7)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)

    _save_plot(fig, path)


def _plot_scatter(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    label_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    xs = []
    ys = []
    labels = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        labels.append(str(row.get(label_key, "")))

    if not xs or not ys:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys)

    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_plot(fig, path)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def _glob_jsonl(dir_path: Path, prefix: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.glob(f"{prefix}*.jsonl") if p.is_file()])


def _load_small_rows(paths: Paths, logger: Any) -> List[Dict[str, Any]]:
    files = _glob_jsonl(paths.small_llm_responses_dir, "SmallLLMResponses")
    rows: List[Dict[str, Any]] = []
    for path in files:
        for row in _read_jsonl(path):
            row = dict(row)
            row["_file"] = path.name
            row["_model_aliases"] = _model_aliases(row)
            row["_small_model_key"] = _primary_model_key(row)
            row["_condition_key"] = _condition_key(row)
            row["_condition_bucket"] = _condition_bucket(row)
            rows.append(row)

    _log(logger, "loaded small model response rows", files=len(files), rows=len(rows))
    return rows


def _load_judge_rows(paths: Paths, logger: Any) -> List[Dict[str, Any]]:
    files = _glob_jsonl(paths.judge_responses_dir, "JudgeLLMResponses")
    rows: List[Dict[str, Any]] = []
    for path in files:
        for row in _read_jsonl(path):
            row = dict(row)
            row["_file"] = path.name
            row["_model_aliases"] = _model_aliases(row)
            row["_small_model_key"] = _primary_model_key(row)
            row["_judge_key"] = _judge_key(row)
            row["_condition_key"] = _condition_key(row)
            row["_condition_bucket"] = _condition_bucket(row)
            rows.append(row)

    _log(logger, "loaded judge response rows", files=len(files), rows=len(rows))
    return rows


def _load_claim_maps(paths: Paths, logger: Any) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    scoring_path = paths.run_claims_for_scoring_jsonl if paths.run_claims_for_scoring_jsonl.exists() else paths.claims_for_scoring_jsonl
    llms_path = paths.run_claims_for_llms_jsonl if paths.run_claims_for_llms_jsonl.exists() else paths.claims_for_llms_jsonl

    scoring_rows = _read_jsonl(scoring_path)
    llm_rows = _read_jsonl(llms_path)

    gold_by_claim: Dict[str, str] = {}
    meta_by_claim: Dict[str, Dict[str, Any]] = {}

    for row in scoring_rows:
        cid = _claim_id(row)
        label = normalize_label(row.get("rating") or row.get("gold_label") or row.get("verdict"))
        if cid is not None and label is not None:
            gold_by_claim[cid] = label

    for row in llm_rows:
        cid = _claim_id(row)
        if cid is None:
            continue
        meta_by_claim[cid] = {
            "claim_text": row.get("claim_text"),
            "lang": row.get("lang"),
            "claim_date": row.get("claim_date"),
        }

    _log(
        logger,
        "loaded claim gold/meta rows",
        gold_claims=len(gold_by_claim),
        claim_meta=len(meta_by_claim),
        scoring_path=str(scoring_path),
        llms_path=str(llms_path),
    )
    return gold_by_claim, meta_by_claim


# ---------------------------------------------------------------------
# Judge merge
# ---------------------------------------------------------------------
def _judge_dim_scores(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    out: Dict[str, Optional[float]] = {}
    for dim in JUDGE_DIMS:
        out[dim] = _as_float(scores.get(dim) if isinstance(scores, dict) else None)
        if out[dim] is None:
            out[dim] = _as_float(row.get(dim))
    return out


def build_per_example_rows(
    small_rows: List[Dict[str, Any]],
    judge_rows: List[Dict[str, Any]],
    gold_by_claim: Dict[str, str],
    meta_by_claim: Dict[str, Dict[str, Any]],
    logger: Any,
) -> List[Dict[str, Any]]:
    # Index judges by (claim_id, condition_key), then match via alias overlap.
    judges_by_claim_condition: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in judge_rows:
        cid = _claim_id(row)
        if cid is None:
            continue
        key = (cid, row["_condition_key"])
        judges_by_claim_condition[key].append(row)

    out: List[Dict[str, Any]] = []

    for row in small_rows:
        cid = _claim_id(row)
        if cid is None:
            _count(logger, "small_rows_missing_claim_id", 1)
            continue

        model_key = row["_small_model_key"]
        condition_key = row["_condition_key"]
        meta = meta_by_claim.get(cid, {})

        candidate_judges = judges_by_claim_condition.get((cid, condition_key), [])
        small_aliases: Set[str] = set(row.get("_model_aliases", []))
        judges: List[Dict[str, Any]] = []

        for j in candidate_judges:
            judge_aliases = set(j.get("_model_aliases", []))
            if small_aliases & judge_aliases:
                judges.append(j)

        judge_overalls = [_as_float(j.get("overall_score")) for j in judges]
        judge_names = sorted({_judge_key(j) for j in judges})

        dim_lists: Dict[str, List[Optional[float]]] = {dim: [] for dim in JUDGE_DIMS}
        for j in judges:
            dims = _judge_dim_scores(j)
            for dim in JUDGE_DIMS:
                dim_lists[dim].append(dims.get(dim))

        pred = normalize_label(row.get("verdict"))
        gold = gold_by_claim.get(cid)

        merged = {
            "claim_id": cid,
            "claim_text": row.get("claim_text") or meta.get("claim_text"),
            "lang": row.get("lang") or meta.get("lang"),
            "claim_date": row.get("claim_date") or meta.get("claim_date"),
            "gold_label": gold,
            "predicted_label": pred,
            "is_correct": (gold == pred) if (gold is not None and pred is not None) else None,
            "small_model_id": row.get("model_id") or row.get("small_model_id"),
            "small_model_name": row.get("model_name") or row.get("small_model_name"),
            "small_model_provider": row.get("provider") or row.get("small_model_provider"),
            "model_key": model_key,
            "model_aliases": sorted(small_aliases),
            "condition": row.get("condition") or condition_key,
            "variant_name": row.get("variant_name"),
            "condition_bucket": row["_condition_bucket"],
            "condition_key": condition_key,
            "corruption_severity": _extract_severity(row.get("variant_name") or condition_key),
            "prompt_id": row.get("prompt_id"),
            "explanation": row.get("explanation"),
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
            "judge_count": len(judges),
            "judge_names": judge_names,
            "judge_overall_sum": sum(v for v in judge_overalls if v is not None),
            "avg_judge_overall": _mean(judge_overalls),
        }

        for dim in JUDGE_DIMS:
            merged[f"avg_{dim}"] = _mean(dim_lists[dim])

        out.append(merged)

    matched = sum(1 for r in out if (r.get("judge_count") or 0) > 0)
    _log(logger, "built per-example rows", rows=len(out), rows_with_judges=matched)
    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def compute_model_metrics(per_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        r for r in per_rows
        if r.get("gold_label") in LABELS and r.get("predicted_label") in LABELS
    ]

    def _report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = [str(r["gold_label"]) for r in rows]
        y_pred = [str(r["predicted_label"]) for r in rows]
        m = classification_metrics(y_true, y_pred, LABELS)
        m["avg_judge_overall"] = _mean(_as_float(r.get("avg_judge_overall")) for r in rows)
        m["n_with_judge"] = sum(1 for r in rows if r.get("avg_judge_overall") is not None)
        return m

    by_model: Dict[str, Any] = {}
    by_model_condition: Dict[str, Any] = {}
    by_model_condition_language: Dict[str, Any] = {}

    for key, rows in _group_by(valid, ["model_key"]).items():
        by_model[str(key[0])] = _report(rows)

    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        by_model_condition[f"{key[0]}||{key[1]}"] = _report(rows)

    for key, rows in _group_by(valid, ["model_key", "condition_key", "lang"]).items():
        by_model_condition_language[f"{key[0]}||{key[1]}||{key[2]}"] = _report(rows)

    return {
        "overall": _report(valid),
        "by_model": by_model,
        "by_model_condition": by_model_condition,
        "by_model_condition_language": by_model_condition_language,
    }


def compute_judge_scores(per_rows: List[Dict[str, Any]], judge_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _judge_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n": len(rows),
            "overall_score_mean": _mean(_as_float(r.get("overall_score")) for r in rows),
            **{
                f"{dim}_mean": _mean(_judge_dim_scores(r).get(dim) for r in rows)
                for dim in JUDGE_DIMS
            },
        }

    overall = _judge_report(judge_rows)

    by_judge: Dict[str, Any] = {}
    for key, rows in _group_by(judge_rows, ["_judge_key"]).items():
        by_judge[str(key[0])] = _judge_report(rows)

    by_small_model: Dict[str, Any] = {}
    by_small_model_condition: Dict[str, Any] = {}
    by_small_model_condition_language: Dict[str, Any] = {}

    for key, rows in _group_by(per_rows, ["model_key"]).items():
        by_small_model[str(key[0])] = {
            "n": len(rows),
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            **{
                f"avg_{dim}": _mean(_as_float(r.get(f"avg_{dim}")) for r in rows)
                for dim in JUDGE_DIMS
            },
        }

    for key, rows in _group_by(per_rows, ["model_key", "condition_key"]).items():
        by_small_model_condition[f"{key[0]}||{key[1]}"] = {
            "n": len(rows),
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            **{
                f"avg_{dim}": _mean(_as_float(r.get(f"avg_{dim}")) for r in rows)
                for dim in JUDGE_DIMS
            },
        }

    for key, rows in _group_by(per_rows, ["model_key", "condition_key", "lang"]).items():
        by_small_model_condition_language[f"{key[0]}||{key[1]}||{key[2]}"] = {
            "n": len(rows),
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            **{
                f"avg_{dim}": _mean(_as_float(r.get(f"avg_{dim}")) for r in rows)
                for dim in JUDGE_DIMS
            },
        }

    agreement = {
        "rows_with_any_judge": sum(1 for r in per_rows if (r.get("judge_count") or 0) > 0),
        "rows_with_2plus_judges": sum(1 for r in per_rows if (r.get("judge_count") or 0) >= 2),
        "avg_judges_per_example": _mean(_as_float(r.get("judge_count")) for r in per_rows),
    }

    return {
        "overall": overall,
        "by_judge": by_judge,
        "by_small_model": by_small_model,
        "by_small_model_condition": by_small_model_condition,
        "by_small_model_condition_language": by_small_model_condition_language,
        "agreement": agreement,
    }


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def build_plots(per_rows: List[Dict[str, Any]], paths: Paths, logger: Any) -> None:
    plots_dir = paths.results_plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    valid = [
        r for r in per_rows
        if r.get("gold_label") in LABELS and r.get("predicted_label") in LABELS
    ]

    # 1) Accuracy by condition
    acc_by_condition = []
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        acc_by_condition.append({
            "model_key": key[0],
            "condition_key": key[1],
            "accuracy": m["accuracy"],
        })

    _safe_plot(
        logger,
        "accuracy_by_condition",
        _plot_grouped_bar,
        acc_by_condition,
        x_key="condition_key",
        series_key="model_key",
        y_key="accuracy",
        title="Accuracy by condition",
        ylabel="Accuracy",
        path=plots_dir / "accuracy_by_condition.png",
    )

    # 2) Accuracy by language
    acc_by_lang = []
    for key, rows in _group_by(valid, ["model_key", "lang"]).items():
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        acc_by_lang.append({
            "model_key": key[0],
            "lang": key[1],
            "accuracy": m["accuracy"],
        })

    _safe_plot(
        logger,
        "accuracy_by_language",
        _plot_grouped_bar,
        acc_by_lang,
        x_key="lang",
        series_key="model_key",
        y_key="accuracy",
        title="Accuracy by language",
        ylabel="Accuracy",
        path=plots_dir / "accuracy_by_language.png",
    )

    # 3) Robustness curve = accuracy drop vs corruption severity
    robustness_rows = []
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        sev = _extract_severity(key[1])
        if sev is None:
            continue
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        robustness_rows.append({
            "model_key": key[0],
            "severity": sev,
            "accuracy": m["accuracy"],
        })

    if robustness_rows:
        _safe_plot(
            logger,
            "robustness_curve_all_models",
            _plot_line,
            robustness_rows,
            x_key="severity",
            series_key="model_key",
            y_key="accuracy",
            title="Robustness curve",
            ylabel="Accuracy",
            path=plots_dir / "robustness_curve_all_models.png",
        )

    # 4) Confusion matrices
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        mat = confusion_matrix_counts(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        safe_name = f"confusion_matrix__{_safe_filename(str(key[0]))}__{_safe_filename(str(key[1]))}"
        _safe_plot(
            logger,
            safe_name,
            _plot_confusion_matrix,
            mat,
            LABELS,
            title=f"Confusion matrix: {key[0]} | {key[1]}",
            path=plots_dir / f"{safe_name}.png",
        )

    # 5) Judge overall by condition
    judge_condition_rows = []
    for key, rows in _group_by(per_rows, ["model_key", "condition_key"]).items():
        judge_condition_rows.append({
            "model_key": key[0],
            "condition_key": key[1],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
        })

    _safe_plot(
        logger,
        "judge_overall_by_condition",
        _plot_grouped_bar,
        judge_condition_rows,
        x_key="condition_key",
        series_key="model_key",
        y_key="avg_judge_overall",
        title="Average judge overall score by condition",
        ylabel="Avg judge overall",
        path=plots_dir / "judge_overall_by_condition.png",
    )

    # 6) Judge heatmap
    model_names = sorted({str(r.get("model_key")) for r in per_rows if r.get("model_key") is not None})
    if model_names:
        matrix: List[List[float]] = []
        for model in model_names:
            rows = [r for r in per_rows if str(r.get("model_key")) == model]
            matrix.append([
                _mean(_as_float(r.get(f"avg_{dim}")) for r in rows) or 0.0
                for dim in JUDGE_DIMS
            ])

        _safe_plot(
            logger,
            "judge_dimension_heatmap",
            _plot_heatmap,
            matrix=matrix,
            row_labels=model_names,
            col_labels=JUDGE_DIMS,
            title="Judge dimension averages by model",
            path=plots_dir / "judge_dimension_heatmap.png",
        )

    # 7) Robustness ranking: mean non-gold accuracy by model
    robustness_rank_rows = []
    for key, rows in _group_by(valid, ["model_key"]).items():
        nongold = [r for r in rows if str(r.get("condition_key")) != "gold"]
        base = nongold if nongold else rows
        m = classification_metrics(
            [r["gold_label"] for r in base],
            [r["predicted_label"] for r in base],
            LABELS,
        )
        robustness_rank_rows.append({
            "rank_bucket": "non_gold_mean_accuracy",
            "model_key": key[0],
            "score": m["accuracy"],
        })

    _safe_plot(
        logger,
        "robustness_ranking",
        _plot_grouped_bar,
        robustness_rank_rows,
        x_key="rank_bucket",
        series_key="model_key",
        y_key="score",
        title="Model robustness ranking",
        ylabel="Mean accuracy on corrupted conditions",
        path=plots_dir / "robustness_ranking.png",
    )

    # 8) Judge vs accuracy scatter
    scatter_rows = []
    for key, rows in _group_by(valid, ["model_key"]).items():
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        scatter_rows.append({
            "model_key": key[0],
            "accuracy": m["accuracy"],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
        })

    _safe_plot(
        logger,
        "judge_vs_accuracy_scatter",
        _plot_scatter,
        scatter_rows,
        x_key="accuracy",
        y_key="avg_judge_overall",
        label_key="model_key",
        title="Judge score vs model accuracy",
        xlabel="Accuracy",
        ylabel="Average judge overall score",
        path=plots_dir / "judge_vs_accuracy_scatter.png",
    )

    _log(logger, "plots written", plots_dir=str(plots_dir))


# ---------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------
def run_step11(
    *,
    paths: Paths,
    config: Any = None,
    run_id: Optional[str] = None,
    code_version: Optional[str] = None,
    config_hash: Optional[str] = None,
    logger: Any = None,
    log: Any = None,
    step_logger: Any = None,
) -> Dict[str, Any]:
    logger = logger or log or step_logger

    paths.ensure_run_dirs()

    gold_by_claim, meta_by_claim = _load_claim_maps(paths, logger)
    small_rows = _load_small_rows(paths, logger)
    judge_rows = _load_judge_rows(paths, logger)

    per_rows = build_per_example_rows(
        small_rows=small_rows,
        judge_rows=judge_rows,
        gold_by_claim=gold_by_claim,
        meta_by_claim=meta_by_claim,
        logger=logger,
    )

    model_metrics = compute_model_metrics(per_rows)
    judge_scores = compute_judge_scores(per_rows, judge_rows)

    _write_json(paths.results_model_metrics_json, model_metrics)
    _write_json(paths.results_judge_scores_json, judge_scores)
    _write_jsonl(paths.results_dir / "per_example_rows.jsonl", per_rows)
    _write_csv(paths.results_dir / "per_example_rows.csv", per_rows)

    build_plots(per_rows, paths, logger)

    summary = {
        "run_id": paths.run_id,
        "n_gold_claims": len(gold_by_claim),
        "n_small_rows": len(small_rows),
        "n_judge_rows": len(judge_rows),
        "n_per_example_rows": len(per_rows),
        "n_rows_with_judges": sum(1 for r in per_rows if (r.get("judge_count") or 0) > 0),
        "results_dir": str(paths.results_dir),
    }

    _log(logger, "step11 finished", **summary)
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    paths = Paths(run_id=args.run_id)
    run_step11(paths=paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())