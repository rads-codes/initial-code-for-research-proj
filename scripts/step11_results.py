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
- runs/<run_id>/results/summary_table_rows.json
- runs/<run_id>/results/summary_table_rows.csv
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
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_COLORS: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "#1B4F8A",
    "qwen/qwen-2.5-7b-instruct": "#808080",
}

MODEL_LABELS: Dict[str, str] = {
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
    "qwen/qwen-2.5-7b-instruct": "Qwen 2.5 7B Instruct",
}

LANG_LABELS: Dict[str, str] = {
    "en": "English",
    "de": "German",
    "el": "Greek",
    "ro": "Romanian",
    "ro_mt_en": "Romanian (Machine-Translated to English)",
}

# Canonical display order for language axes.  Languages not in this list are
# appended at the end in sorted order.  ro_mt_en is placed immediately after ro
# so the comparison pair appears side-by-side in every chart.
LANG_ORDER: List[str] = ["en", "de", "el", "ro", "ro_mt_en"]


def _lang_sort_key(lang: str) -> Tuple[int, str]:
    try:
        return (LANG_ORDER.index(lang), lang)
    except ValueError:
        return (len(LANG_ORDER), lang)

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
    aliases: List[str] = []
    for key in ("small_model_id", "model_id", "small_model_name", "model_name"):
        v = _s(row.get(key))
        if v and v not in aliases:
            aliases.append(v)
    return aliases


def _primary_model_key(row: Dict[str, Any]) -> str:
    return (
        _s(row.get("small_model_name"))
        or _s(row.get("model_name"))
        or _s(row.get("small_model_id"))
        or _s(row.get("model_id"))
        or "unknown_model"
    )


def _judge_key(row: Dict[str, Any]) -> str:
    return (
        _s(row.get("judge_model_name"))
        or _s(row.get("judge_name"))
        or _s(row.get("judge_id"))
        or "unknown_judge"
    )


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


def _extract_corruption_family(condition_or_variant: Optional[str]) -> Optional[str]:
    if not condition_or_variant:
        return "gold"
    s = str(condition_or_variant).lower()
    if s == "gold":
        return "gold"
    if "target" in s:
        return "targeted"
    if "random" in s and ("drop" in s or "remove" in s):
        return "random"
    if "replace" in s:
        return "replace"
    return "other"


def _condition_sort_key(val: Any) -> Tuple[int, int, float, str]:
    s = str(val)
    if s == "gold":
        return (0, 0, 0.0, s)

    low = s.lower()
    if "target" in low:
        method_rank = 1
    elif "random" in low and ("drop" in low or "remove" in low):
        method_rank = 2
    elif "replace" in low or "mix" in low:
        method_rank = 3
    else:
        method_rank = 9

    sev = _extract_severity(s)
    sev_num = float(sev) if sev is not None else 999.0
    return (1, method_rank, sev_num, s)


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


def confusion_matrix_row_normalized(
    y_true: List[str],
    y_pred: List[str],
    labels: Sequence[str],
) -> List[List[float]]:
    counts = confusion_matrix_counts(y_true, y_pred, labels)
    out: List[List[float]] = []
    for row in counts:
        s = sum(row)
        if s == 0:
            out.append([0.0 for _ in row])
        else:
            out.append([v / s for v in row])
    return out


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
def _fmt_label(s: str) -> str:
    """Format a snake_case key for display: title-case words, append % to trailing numbers."""
    key = str(s).lower()
    if key in LANG_LABELS:
        return LANG_LABELS[key]
    parts = str(s).split("_")
    out = []
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d+", p) and i == len(parts) - 1:
            out.append((p.lstrip("0") or "0") + "%")
        else:
            out.append(p.capitalize())
    return " ".join(out)


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
    x_order: Optional[Sequence[str]] = None,
    n_key: Optional[str] = None,
    value_fmt: str = "{:.3f}",
    xlabel: Optional[str] = None,
) -> None:
    if x_order is not None:
        xs = [str(x) for x in x_order if any(str(r.get(x_key)) == str(x) for r in rows)]
    else:
        xs = sorted({str(r[x_key]) for r in rows if r.get(x_key) is not None})

    series = sorted({str(r[series_key]) for r in rows if r.get(series_key) is not None})
    if not xs or not series:
        return

    lookup = {(str(r[x_key]), str(r[series_key])): r for r in rows}
    width = 0.8 / max(1, len(series))

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(xs)))

    for si, s in enumerate(series):
        x_positions = []
        ys = []
        for i, x in enumerate(xs):
            x_positions.append(i - 0.4 + width / 2 + si * width)
            row = lookup.get((x, s), {})
            y = row.get(y_key)
            y_val = 0.0 if y is None else float(y)
            ys.append(y_val)

        fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        color = MODEL_COLORS.get(s, fallback_colors[si % len(fallback_colors)])
        ax.bar(
            x_positions,
            ys,
            width=width,
            label=MODEL_LABELS.get(s, s),
            color=color,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([_fmt_label(x) for x in xs], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel or _fmt_label(x_key))
    ax.set_ylim(bottom=0)
    ax.legend()
    _save_plot(fig, path)


def _plot_nested_grouped_bar(
    rows: List[Dict[str, Any]],
    outer_key: str,
    inner_key: str,
    series_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    path: Path,
    outer_order: Optional[Sequence[str]] = None,
    inner_order: Optional[Sequence[str]] = None,
    series_order: Optional[Sequence[str]] = None,
    n_key: Optional[str] = None,
    value_fmt: str = "{:.3f}",
    xlabel: Optional[str] = None,
) -> None:
    """
    Three-level grouped bar chart:
      outer_key  -> major group (e.g. language)
      inner_key  -> subgroup inside each major group (e.g. condition)
      series_key -> bars inside each subgroup (e.g. model)
    """
    if outer_order is not None:
        outers = [str(x) for x in outer_order if any(str(r.get(outer_key)) == str(x) for r in rows)]
    else:
        outers = sorted({str(r[outer_key]) for r in rows if r.get(outer_key) is not None})

    if inner_order is not None:
        inners = [str(x) for x in inner_order if any(str(r.get(inner_key)) == str(x) for r in rows)]
    else:
        inners = sorted({str(r[inner_key]) for r in rows if r.get(inner_key) is not None})

    if series_order is not None:
        series = [str(x) for x in series_order if any(str(r.get(series_key)) == str(x) for r in rows)]
    else:
        series = sorted({str(r[series_key]) for r in rows if r.get(series_key) is not None})

    if not outers or not inners or not series:
        return

    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    lookup = {
        (str(r.get(outer_key)), str(r.get(inner_key)), str(r.get(series_key))): r
        for r in rows
        if r.get(outer_key) is not None and r.get(inner_key) is not None and r.get(series_key) is not None
    }

    n_outer = len(outers)
    n_inner = len(inners)
    n_series = len(series)

    subgroup_width = 0.84
    bar_width = subgroup_width / max(1, n_series)
    outer_gap = 0.9

    fig_w = max(12, n_outer * n_inner * 1.25)
    fig, ax = plt.subplots(figsize=(fig_w, 6.5))

    subgroup_centers: List[float] = []
    subgroup_labels: List[str] = []
    outer_centers: List[float] = []

    for oi, outer_val in enumerate(outers):
        block_start = oi * (n_inner + outer_gap)

        for ii, inner_val in enumerate(inners):
            subgroup_left = block_start + ii
            subgroup_center = subgroup_left + 0.5
            subgroup_centers.append(subgroup_center)
            subgroup_labels.append(_fmt_label(inner_val))

            for si, series_val in enumerate(series):
                row = lookup.get((outer_val, inner_val, series_val), {})
                y = row.get(y_key)
                y_val = 0.0 if y is None else float(y)

                x = subgroup_left + (si + 0.5) * bar_width
                display_label = MODEL_LABELS.get(series_val, series_val)
                label = display_label if (oi == 0 and ii == 0) else None
                color = MODEL_COLORS.get(series_val, fallback_colors[si % len(fallback_colors)])
                ax.bar(
                    x,
                    y_val,
                    width=bar_width,
                    label=label,
                    color=color,
                )

        outer_center = block_start + (n_inner - 1) / 2 + 0.5
        outer_centers.append(outer_center)

    ax.set_xticks(subgroup_centers)
    ax.set_xticklabels(subgroup_labels, rotation=35, ha="right")

    for center, outer_val in zip(outer_centers, outers):
        ax.text(
            center,
            -0.22,
            _fmt_label(outer_val),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    for oi in range(1, n_outer):
        sep_x = oi * (n_inner + outer_gap) - (outer_gap / 2)
        ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel or f"{_fmt_label(inner_key)} grouped by {_fmt_label(outer_key)}")
    ax.set_ylim(bottom=0)
    ax.legend()
    _save_plot(fig, path)


def _plot_points(
    rows: List[Dict[str, Any]],
    x_key: str,
    series_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    path: Path,
    xlabel: Optional[str] = None,
    n_key: Optional[str] = None,
    value_fmt: str = "{:.3f}",
) -> None:
    series = sorted({str(r[series_key]) for r in rows if r.get(series_key) is not None})
    xs = sorted({float(r[x_key]) for r in rows if r.get(x_key) is not None})
    if not series or not xs:
        return

    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_map = {s: MODEL_COLORS.get(s, fallback_colors[i % len(fallback_colors)]) for i, s in enumerate(series)}

    fig, ax = plt.subplots(figsize=(10, 5))

    for s in series:
        color = color_map[s]
        display = MODEL_LABELS.get(s, s)
        subset = [r for r in rows if str(r.get(series_key)) == s]
        for row in subset:
            x = row.get(x_key)
            y = row.get(y_key)
            if x is None or y is None:
                continue
            x = float(x)
            y = float(y)
            ax.scatter([x], [y], label=display, color=color)

    handles, labels = ax.get_legend_handles_labels()
    seen: Set[str] = set()
    dedup_handles = []
    dedup_labels = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            dedup_handles.append(h)
            dedup_labels.append(l)

    ax.set_title(title)
    ax.set_xlabel(xlabel or _fmt_label(x_key))
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(dedup_handles, dedup_labels, fontsize=8)
    _save_plot(fig, path)


def _plot_confusion_matrix(mat: List[List[int]], labels: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap="Blues_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    fmt_labels = [_fmt_label(l) for l in labels]
    ax.set_xticklabels(fmt_labels, rotation=30, ha="right")
    ax.set_yticklabels(fmt_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    flat = [mat[i][j] for i in range(len(labels)) for j in range(len(labels))]
    vmin, vmax = min(flat), max(flat)
    for i in range(len(labels)):
        for j in range(len(labels)):
            norm = (mat[i][j] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "white" if norm < 0.5 else "black"
            ax.text(j, i, str(mat[i][j]), ha="center", va="center", fontsize=8, color=color)

    _save_plot(fig, path)


def _plot_confusion_matrix_float(mat: List[List[float]], labels: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap="Blues_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    fmt_labels = [_fmt_label(l) for l in labels]
    ax.set_xticklabels(fmt_labels, rotation=30, ha="right")
    ax.set_yticklabels(fmt_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if mat[i][j] < 0.5 else "black"
            ax.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=8, color=color)

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
    im = ax.imshow(matrix, aspect="auto", cmap="Blues_r")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([_fmt_label(l) for l in col_labels], rotation=30, ha="right")
    ax.set_yticklabels(list(row_labels))
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    flat_h = [matrix[i][j] for i in range(len(row_labels)) for j in range(len(col_labels))]
    h_vmin, h_vmax = min(flat_h), max(flat_h)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            norm = (matrix[i][j] - h_vmin) / (h_vmax - h_vmin) if h_vmax > h_vmin else 0.5
            text_color = "white" if norm < 0.5 else "black"
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8, color=text_color)

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
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        color = MODEL_COLORS.get(lab, fallback_colors[i % len(fallback_colors)])
        display = MODEL_LABELS.get(lab, lab)
        ax.scatter([x], [y], color=color, zorder=5)
        ax.annotate(display, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=25.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_plot(fig, path)


def _plot_table(rows: List[Dict[str, Any]], columns: Sequence[str], title: str, path: Path) -> None:
    if not rows or not columns:
        return

    cell_text = []
    for row in rows:
        out_row = []
        for col in columns:
            v = row.get(col)
            if isinstance(v, float):
                out_row.append(f"{v:.2f}")
            elif v is None:
                out_row.append("")
            else:
                out_row.append(str(v))
        cell_text.append(out_row)

    fig_h = max(3, 0.45 * (len(rows) + 2))
    fig_w = max(10, 1.4 * len(columns))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=[_fmt_label(c) for c in columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(columns))))
    if "n" in list(columns):
        n_col_idx = list(columns).index("n")
        for row_idx in range(0, len(rows) + 1):
            cell = table[row_idx, n_col_idx]
            cell.set_width(cell.get_width() * 1.8)
    table.scale(1, 1.35)
    ax.set_title(title, pad=12)
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

        condition_or_variant = row.get("variant_name") or condition_key

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
            "corruption_severity": _extract_severity(condition_or_variant),
            "corruption_family": _extract_corruption_family(condition_or_variant),
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


def build_summary_table_rows(per_rows: List[Dict[str, Any]], judge_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [
        r for r in per_rows
        if r.get("gold_label") in LABELS and r.get("predicted_label") in LABELS
    ]

    judge_names_all = sorted({_judge_key(r) for r in judge_rows if _judge_key(r)})
    judge_names_str = ", ".join(judge_names_all)

    out: List[Dict[str, Any]] = []
    grouped = _group_by(valid, ["model_key", "condition_key"])

    for key, rows in grouped.items():
        model_key = str(key[0])
        condition_key = str(key[1])

        y_true = [str(r["gold_label"]) for r in rows]
        y_pred = [str(r["predicted_label"]) for r in rows]
        m = classification_metrics(y_true, y_pred, LABELS)

        langs = sorted({str(r.get("lang")) for r in rows if r.get("lang") is not None})
        per_row_judges = sorted({j for r in rows for j in (r.get("judge_names") or [])})

        severity_vals = sorted({
            r.get("corruption_severity")
            for r in rows
            if r.get("corruption_severity") is not None
        })
        family_vals = sorted({
            _extract_corruption_family(r.get("variant_name") or r.get("condition_key"))
            for r in rows
            if (r.get("variant_name") or r.get("condition_key")) is not None
        })

        summary_row = {
            "model_key": model_key,
            "condition_key": condition_key,
            "condition_bucket": rows[0].get("condition_bucket"),
            "corruption_family": ", ".join(family_vals),
            "severity_values": severity_vals,
            "primary_severity": severity_vals[0] if severity_vals else None,
            "languages": ", ".join(langs),
            "language_count": len(langs),
            "judges_seen": ", ".join(per_row_judges) if per_row_judges else judge_names_str,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            "avg_political_bias": _mean(_as_float(r.get("avg_political_bias")) for r in rows),
            "avg_sociocultural_bias": _mean(_as_float(r.get("avg_sociocultural_bias")) for r in rows),
            "avg_linguistic_bias": _mean(_as_float(r.get("avg_linguistic_bias")) for r in rows),
            "avg_logic_of_reasoning": _mean(_as_float(r.get("avg_logic_of_reasoning")) for r in rows),
            "avg_evidence_usage": _mean(_as_float(r.get("avg_evidence_usage")) for r in rows),
        }
        out.append(summary_row)

    out.sort(key=lambda r: (_condition_sort_key(r["condition_key"]), str(r["model_key"])))
    return out

def build_language_summary_table_rows(per_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [
        r for r in per_rows
        if r.get("gold_label") in LABELS and r.get("predicted_label") in LABELS
    ]

    out: List[Dict[str, Any]] = []

    grouped = _group_by(valid, ["model_key", "lang"])

    for key, rows in grouped.items():
        model_key = str(key[0])
        lang = str(key[1])

        y_true = [r["gold_label"] for r in rows]
        y_pred = [r["predicted_label"] for r in rows]

        m = classification_metrics(y_true, y_pred, LABELS)

        summary_row = {
            "model_key": model_key,
            "language": lang,
            "n": m["n"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            "avg_political_bias": _mean(_as_float(r.get("avg_political_bias")) for r in rows),
            "avg_sociocultural_bias": _mean(_as_float(r.get("avg_sociocultural_bias")) for r in rows),
            "avg_linguistic_bias": _mean(_as_float(r.get("avg_linguistic_bias")) for r in rows),
            "avg_logic_of_reasoning": _mean(_as_float(r.get("avg_logic_of_reasoning")) for r in rows),
            "avg_evidence_usage": _mean(_as_float(r.get("avg_evidence_usage")) for r in rows),
        }

        out.append(summary_row)

    out.sort(key=lambda r: (_lang_sort_key(r["language"]), r["model_key"]))
    return out

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def build_plots(
    per_rows: List[Dict[str, Any]],
    judge_rows: List[Dict[str, Any]],
    summary_table_rows: List[Dict[str, Any]],
    language_summary_rows: List[Dict[str, Any]],
    paths: Paths,
    logger: Any,
) -> None:
    plots_dir = paths.results_plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    valid = [
        r for r in per_rows
        if r.get("gold_label") in LABELS and r.get("predicted_label") in LABELS
    ]

    condition_order = sorted(
        {str(r.get("condition_key")) for r in per_rows if r.get("condition_key") is not None},
        key=_condition_sort_key,
    )
    language_order = sorted(
        {str(r.get("lang")) for r in per_rows if r.get("lang") is not None},
        key=_lang_sort_key,
    )
    model_order = sorted(
        {str(r.get("model_key")) for r in per_rows if r.get("model_key") is not None}
    )

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
            "n": m["n"],
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
        x_order=condition_order,
        n_key="n",
        xlabel="Corruption Level",
    )

    _safe_plot(
        logger,
        "language_summary_table",
        _plot_table,
        language_summary_rows,
        columns=[
            "model_key",
            "language",
            "n",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "avg_judge_overall",
        ],
        title="Experiment summary by language",
        path=plots_dir / "language_summary_table.png",
    )

    # 2) Average judge overall by condition
    judge_by_condition = []
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        judge_by_condition.append({
            "model_key": key[0],
            "condition_key": key[1],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            "n": len(rows),
        })

    _safe_plot(
        logger,
        "avg_judge_overall_by_condition",
        _plot_grouped_bar,
        judge_by_condition,
        x_key="condition_key",
        series_key="model_key",
        y_key="avg_judge_overall",
        title="Average judge overall score by condition",
        ylabel="Avg judge overall",
        path=plots_dir / "avg_judge_overall_by_condition.png",
        x_order=condition_order,
        n_key="n",
        xlabel="Corruption Level",
    )

    # 3) Accuracy by language (gold evidence only)
    gold_only = [r for r in valid if str(r.get("condition_key")) == "gold"]
    acc_by_lang = []
    for key, rows in _group_by(gold_only, ["model_key", "lang"]).items():
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        acc_by_lang.append({
            "model_key": key[0],
            "lang": key[1],
            "accuracy": m["accuracy"],
            "n": m["n"],
        })

    _safe_plot(
        logger,
        "accuracy_by_language",
        _plot_grouped_bar,
        acc_by_lang,
        x_key="lang",
        series_key="model_key",
        y_key="accuracy",
        title="Accuracy by Language on Gold Evidence",
        ylabel="Accuracy",
        path=plots_dir / "accuracy_by_language.png",
        x_order=language_order,
        n_key="n",
        xlabel="Language",
    )

    # 3b) Accuracy by language and condition
    acc_by_lang_condition = []
    for key, rows in _group_by(valid, ["lang", "condition_key", "model_key"]).items():
        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        acc_by_lang_condition.append({
            "lang": key[0],
            "condition_key": key[1],
            "model_key": key[2],
            "accuracy": m["accuracy"],
            "n": m["n"],
        })

    _safe_plot(
        logger,
        "accuracy_by_language_and_condition",
        _plot_nested_grouped_bar,
        acc_by_lang_condition,
        outer_key="lang",
        inner_key="condition_key",
        series_key="model_key",
        y_key="accuracy",
        title="Accuracy by language and condition",
        ylabel="Accuracy",
        path=plots_dir / "accuracy_by_language_and_condition.png",
        outer_order=language_order,
        inner_order=condition_order,
        series_order=model_order,
        n_key="n",
        xlabel="Corruption Conditions, Grouped by Language",
    )

    # 4) Robustness points by corruption family
    robustness_rows = []
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        condition_key = str(key[1])
        sev = _extract_severity(condition_key)
        fam = _extract_corruption_family(condition_key)
        if sev is None or fam == "gold":
            continue

        m = classification_metrics(
            [r["gold_label"] for r in rows],
            [r["predicted_label"] for r in rows],
            LABELS,
        )
        robustness_rows.append({
            "model_key": key[0],
            "condition_key": condition_key,
            "corruption_family": fam,
            "severity": sev,
            "accuracy": m["accuracy"],
            "n": m["n"],
        })

    targeted_rows = [r for r in robustness_rows if r.get("corruption_family") == "targeted"]
    random_rows = [r for r in robustness_rows if r.get("corruption_family") == "random"]

    if targeted_rows:
        _safe_plot(
            logger,
            "robustness_points_targeted",
            _plot_points,
            targeted_rows,
            x_key="severity",
            series_key="model_key",
            y_key="accuracy",
            title="Robustness under targeted corruption",
            ylabel="Accuracy",
            path=plots_dir / "robustness_points_targeted.png",
            xlabel="Severity",
            n_key="n",
        )

    if random_rows:
        _safe_plot(
            logger,
            "robustness_points_random",
            _plot_points,
            random_rows,
            x_key="severity",
            series_key="model_key",
            y_key="accuracy",
            title="Robustness under random corruption",
            ylabel="Accuracy",
            path=plots_dir / "robustness_points_random.png",
            xlabel="Severity",
            n_key="n",
        )

    # 5) Confusion matrices: counts + row-normalized
    for key, rows in _group_by(valid, ["model_key", "condition_key"]).items():
        y_true = [r["gold_label"] for r in rows]
        y_pred = [r["predicted_label"] for r in rows]

        mat_counts = confusion_matrix_counts(y_true, y_pred, LABELS)
        mat_norm = confusion_matrix_row_normalized(y_true, y_pred, LABELS)

        safe_name = f"confusion_matrix__{_safe_filename(str(key[0]))}__{_safe_filename(str(key[1]))}"

        model_display = MODEL_LABELS.get(str(key[0]), str(key[0]))
        condition_display = _fmt_label(str(key[1]))

        _safe_plot(
            logger,
            safe_name,
            _plot_confusion_matrix,
            mat_counts,
            LABELS,
            title=f"Confusion Matrix for {model_display} on {condition_display}",
            path=plots_dir / f"{safe_name}.png",
        )

        _safe_plot(
            logger,
            f"{safe_name}__row_normalized",
            _plot_confusion_matrix_float,
            mat_norm,
            LABELS,
            title=f"Row-Normalized Confusion Matrix for {model_display} on {condition_display}",
            path=plots_dir / f"{safe_name}__row_normalized.png",
        )

    # 6) Judge dimension heatmap by model
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
            row_labels=[MODEL_LABELS.get(m, m) for m in model_names],
            col_labels=JUDGE_DIMS,
            title="Judge dimension averages by model",
            path=plots_dir / "judge_dimension_heatmap.png",
        )

    # 7) Judge dimensions by condition
    judge_dim_condition_rows = []
    for key, rows in _group_by(per_rows, ["model_key", "condition_key"]).items():
        for dim in JUDGE_DIMS:
            judge_dim_condition_rows.append({
                "model_key": key[0],
                "condition_key": key[1],
                "dimension": dim,
                "score": _mean(_as_float(r.get(f"avg_{dim}")) for r in rows),
                "n": len(rows),
            })

    for dim in JUDGE_DIMS:
        dim_rows = [r for r in judge_dim_condition_rows if r["dimension"] == dim]
        _safe_plot(
            logger,
            f"{dim}_by_condition",
            _plot_grouped_bar,
            dim_rows,
            x_key="condition_key",
            series_key="model_key",
            y_key="score",
            title=f"{_fmt_label(dim)} by condition",
            ylabel=_fmt_label(dim),
            path=plots_dir / f"{_safe_filename(dim)}_by_condition.png",
            x_order=condition_order,
            n_key="n",
            xlabel="Corruption Level",
        )

    # 8) Robustness ranking: mean non-gold accuracy by model
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
            "n": m["n"],
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
        n_key="n",
    )

    # 9) Judge vs accuracy scatter
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

    # 10) Avg judge overall by language
    judge_by_lang_rows = []
    for key, rows in _group_by(per_rows, ["model_key", "lang"]).items():
        judge_by_lang_rows.append({
            "model_key": key[0],
            "lang": key[1],
            "avg_judge_overall": _mean(_as_float(r.get("avg_judge_overall")) for r in rows),
            "n": len(rows),
        })

    _safe_plot(
        logger,
        "avg_judge_overall_by_language",
        _plot_grouped_bar,
        judge_by_lang_rows,
        x_key="lang",
        series_key="model_key",
        y_key="avg_judge_overall",
        title="Average judge overall score by language",
        ylabel="Avg judge overall",
        path=plots_dir / "avg_judge_overall_by_language.png",
        x_order=language_order,
        n_key="n",
        xlabel="Language",
    )

    # 11) Summary table plot
    summary_columns = [
        "model_key",
        "condition_key",
        "corruption_family",
        "primary_severity",
        "n",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "avg_judge_overall",
    ]
    _safe_plot(
        logger,
        "summary_table",
        _plot_table,
        summary_table_rows,
        columns=summary_columns,
        title="Experiment summary",
        path=plots_dir / "experiment_summary_table.png",
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
    summary_table_rows = build_summary_table_rows(per_rows, judge_rows)
    language_summary_rows = build_language_summary_table_rows(per_rows)
    _write_json(paths.results_model_metrics_json, model_metrics)
    _write_json(paths.results_judge_scores_json, judge_scores)
    _write_jsonl(paths.results_dir / "per_example_rows.jsonl", per_rows)
    _write_csv(paths.results_dir / "per_example_rows.csv", per_rows)

    _write_json(paths.results_dir / "summary_table_rows.json", summary_table_rows)
    _write_csv(paths.results_dir / "summary_table_rows.csv", summary_table_rows)
    _write_json(paths.results_dir / "language_summary_rows.json", language_summary_rows)
    _write_csv(paths.results_dir / "language_summary_rows.csv", language_summary_rows)

    build_plots(
        per_rows=per_rows,
        judge_rows=judge_rows,
        summary_table_rows=summary_table_rows,
        language_summary_rows=language_summary_rows,
        paths=paths,
        logger=logger,
    )

    summary = {
        "run_id": paths.run_id,
        "n_gold_claims": len(gold_by_claim),
        "n_small_rows": len(small_rows),
        "n_judge_rows": len(judge_rows),
        "n_per_example_rows": len(per_rows),
        "n_rows_with_judges": sum(1 for r in per_rows if (r.get("judge_count") or 0) > 0),
        "n_summary_rows": len(summary_table_rows),
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
    main()