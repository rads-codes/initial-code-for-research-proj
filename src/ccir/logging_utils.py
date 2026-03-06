from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from ccir.io_utils import append_jsonl
from ccir.schemas import utc_now_iso


'''
Creates per-step report_<##>.jsonl and provides consistent counters/timers
'''
"""
src/ccir/logging_utils.py

Creates per-step report_<##>.jsonl and provides consistent counters/timers/log rows.

Docs:
- "logging_utils.py: Creates per-step report_<##>.jsonl and provides consistent counters/timers"
- Per-step report files live under: runs/<run_id>/reports/report_00.jsonl (via Paths.report_jsonl)

Design goals:
- Tiny, dependency-light.
- Append-only JSONL (durable via io_utils.append_jsonl).
- Low ceremony: usable from step scripts and __main__ orchestration.
- Does NOT write run_metadata.jsonl (that is handled by utils/run_metadata.py).
"""

JsonObj = Dict[str, Any]


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion for dataclasses and common types."""
    if is_dataclass(x):
        return asdict(x)
    return x


class StepLogger:
    """
    Logger for a single pipeline step that writes JSONL rows to report_<##>.jsonl.

    Typical usage inside a step script:
        logger = StepLogger(paths.report_jsonl(0), run_id, code_version, step="00_prepare_dataset")
        logger.start()
        logger.incr("rows_read", 123)
        with logger.timer("normalize"):
            ...
        logger.summary()

    Prefer the context manager `step_logger(...)` for automatic error+summary handling.
    """

    def __init__(
        self,
        report_path: Path,
        *,
        run_id: str,
        code_version: str,
        step: str,
    ) -> None:
        self.report_path = report_path
        self.run_id = run_id
        self.code_version = code_version
        self.step = step

        self.counts: Dict[str, int] = {}
        self.metrics: Dict[str, float] = {}
        self.timers_s: Dict[str, float] = {}

        self._t0 = time.perf_counter()
        self._started = False
        self._ended = False

    def _base_row(self, event: str) -> JsonObj:
        return {
            "run_id": self.run_id,
            "created_utc": utc_now_iso(),
            "code_version": self.code_version,
            "step": self.step,
            "event": event,
        }

    def log(self, event: str, *, message: Optional[str] = None, **fields: Any) -> None:
        """
        Append a log row to the report JSONL.

        event: short string category (e.g., start/end/info/count/timer/error/summary)
        message: optional human-readable string
        fields: extra structured data (must be JSON-serializable)
        """
        row = self._base_row(event)
        if message is not None:
            row["message"] = message
        for k, v in fields.items():
            row[k] = _to_jsonable(v)
        append_jsonl(self.report_path, row)

    def start(self, *, message: Optional[str] = None, **fields: Any) -> None:
        if self._started:
            return
        self._started = True
        self.log("start", message=message, **fields)

    def incr(self, name: str, n: int = 1, *, emit: bool = False) -> int:
        """
        Increment an integer counter.

        If emit=True, also write a 'count' row (can be noisy; default False).
        """
        if not name:
            raise ValueError("counter name must be non-empty")
        self.counts[name] = int(self.counts.get(name, 0) + n)
        if emit:
            self.log("count", counter=name, value=self.counts[name], delta=n)
        return self.counts[name]

    def set_metric(self, name: str, value: float, *, emit: bool = False) -> None:
        """
        Record a float-ish metric (last value wins unless you store arrays yourself).
        """
        if not name:
            raise ValueError("metric name must be non-empty")
        self.metrics[name] = float(value)
        if emit:
            self.log("metric", metric=name, value=self.metrics[name])

    @contextmanager
    def timer(self, name: str, *, emit: bool = True) -> Iterator[None]:
        """
        Time a block. Accumulates total seconds under timers_s[name].

        If emit=True (default), writes a 'timer' row for this block.
        """
        if not name:
            raise ValueError("timer name must be non-empty")
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.timers_s[name] = float(self.timers_s.get(name, 0.0) + dt)
            if emit:
                self.log("timer", timer=name, elapsed_s=dt, total_s=self.timers_s[name])

    def error(self, exc: BaseException, *, message: Optional[str] = None) -> None:
        """
        Log an error row with structured exception info.
        """
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log(
            "error",
            message=message or str(exc),
            error_type=type(exc).__name__,
            error_message=str(exc),
            trace=trace,
        )

    def summary(self, *, status: str = "ok", message: Optional[str] = None, **fields: Any) -> None:
        """
        Write a final summary row with accumulated counts/metrics/timers and total elapsed.
        """
        if self._ended:
            return
        self._ended = True
        total_elapsed = time.perf_counter() - self._t0
        row = {
            **self._base_row("summary"),
            "status": status,
            "elapsed_s": float(total_elapsed),
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
            "timers_s": dict(self.timers_s),
        }
        if message is not None:
            row["message"] = message
        for k, v in fields.items():
            row[k] = _to_jsonable(v)
        append_jsonl(self.report_path, row)


@contextmanager
def step_logger(
    report_path: Path,
    *,
    run_id: str,
    code_version: str,
    step: str,
    start_message: Optional[str] = None,
    start_fields: Optional[Dict[str, Any]] = None,
) -> Iterator[StepLogger]:
    """
    Convenience context manager:
    - writes start row
    - yields logger
    - on exception: writes error + summary(status="error") then re-raises
    - on success: writes summary(status="ok")

    Example:
        with step_logger(paths.report_jsonl(0), run_id=..., code_version=..., step="00_prepare_dataset") as log:
            ...
            log.incr("kept", 123)
    """
    logger = StepLogger(report_path, run_id=run_id, code_version=code_version, step=step)
    logger.start(message=start_message, **(start_fields or {}))
    try:
        yield logger
    except Exception as e:
        logger.error(e)
        logger.summary(status="error")
        raise
    else:
        logger.summary(status="ok")

#to use this
'''
from ccir.logging_utils import step_logger

def main(paths, run_id: str, code_version: str) -> None:
    report_path = paths.report_jsonl(0)  # report_00.jsonl under runs/<run_id>/reports/
    with step_logger(report_path, run_id=run_id, code_version=code_version, step="00_prepare_dataset") as log:
        log.incr("rows_read", 100, emit=False)
        with log.timer("normalize_dates"):
            ...
        log.incr("rows_written", 95)
'''