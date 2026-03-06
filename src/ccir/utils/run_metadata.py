from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union, ContextManager
import traceback
from contextlib import contextmanager

from ccir.io_utils import append_jsonl, ensure_parent_dir
from ccir.schemas import (
    MetadataError,
    MetadataEvents,
    MetadataEventType,
    to_dict,
    utc_now_iso,
)

try:
    # Optional: lets you pass Paths directly without importing it everywhere.
    from ccir.paths import Paths  # type: ignore
except Exception:  # pragma: no cover
    Paths = None  # type: ignore


class RunMetadataLogger:
    """
    Writes MetadataEvents rows to runs/<run_id>/run_metadata.jsonl.

    Required by schemas.Lineage on every event:
      - run_id
      - created_utc
      - code_version
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        run_metadata_path: Optional[Path] = None,
        code_version: str,
        config_hash: Optional[str] = None,
        paths: Optional["Paths"] = None,
    ) -> None:
        """
        Preferred usage:
          RunMetadataLogger(paths=paths, code_version=..., config_hash=...)

        Backward-compatible usage:
          RunMetadataLogger(run_id=..., run_metadata_path=..., code_version=..., config_hash=...)
        """
        if paths is not None:
            # Use Paths as source of truth
            self.run_id = paths.run_id
            self.run_metadata_path = Path(paths.run_metadata_jsonl)
        else:
            if not run_id:
                raise ValueError("run_id is required when paths is not provided")
            if run_metadata_path is None:
                raise ValueError("run_metadata_path is required when paths is not provided")
            self.run_id = run_id
            self.run_metadata_path = Path(run_metadata_path)

        if not code_version:
            raise ValueError("code_version must be a non-empty string")

        self.code_version = code_version
        self.config_hash = config_hash  # typically logged on run_start

    def _append(self, event: MetadataEvents) -> None:
        ensure_parent_dir(self.run_metadata_path)
        append_jsonl(self.run_metadata_path, to_dict(event))

    @staticmethod
    def _validate_counts(counts: Optional[Dict[str, int]]) -> None:
        if counts is None:
            return
        for k, v in counts.items():
            if not isinstance(k, str):
                raise TypeError(f"counts keys must be str, got {type(k)}")
            if not isinstance(v, int):
                raise TypeError(f"counts[{k!r}] must be int, got {type(v)}")

    @staticmethod
    def _validate_metrics(metrics: Optional[Dict[str, float]]) -> None:
        if metrics is None:
            return
        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError(f"metrics keys must be str, got {type(k)}")
            if not isinstance(v, (int, float)):
                raise TypeError(f"metrics[{k!r}] must be float-ish, got {type(v)}")

    def _make(
        self,
        *,
        event: MetadataEventType,
        step: Optional[str] = None,
        config_hash: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[MetadataError] = None,
    ) -> MetadataEvents:
        self._validate_counts(counts)
        self._validate_metrics(metrics)

        return MetadataEvents(
            run_id=self.run_id,
            created_utc=utc_now_iso(),
            code_version=self.code_version,
            event=event,
            step=step,
            config_hash=config_hash,
            inputs=inputs,
            outputs=outputs,
            counts=counts,
            metrics=metrics,
            error=error,
        )

    @staticmethod
    def _exc_to_error(exc: BaseException) -> MetadataError:
        return MetadataError(
            type=exc.__class__.__name__,
            message=str(exc),
            trace="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    # -----------------------------
    # Public helpers
    # -----------------------------

    def log_run_start(
        self,
        *,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self._append(
            self._make(
                event="run_start",
                config_hash=self.config_hash,
                inputs=inputs,
                outputs=outputs,
                counts=counts,
                metrics=metrics,
            )
        )

    def log_step_start(
        self,
        *,
        step: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not step:
            raise ValueError("step must be a non-empty string")
        self._append(
            self._make(
                event="step_start",
                step=step,
                inputs=inputs,
                outputs=outputs,
                counts=counts,
                metrics=metrics,
            )
        )

    def log_step_end(
        self,
        *,
        step: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not step:
            raise ValueError("step must be a non-empty string")
        self._append(
            self._make(
                event="step_end",
                step=step,
                inputs=inputs,
                outputs=outputs,
                counts=counts,
                metrics=metrics,
            )
        )

    def log_step_error(
        self,
        *,
        step: str,
        exc: BaseException,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not step:
            raise ValueError("step must be a non-empty string")
        self._append(
            self._make(
                event="step_error",
                step=step,
                inputs=inputs,
                outputs=outputs,
                counts=counts,
                metrics=metrics,
                error=self._exc_to_error(exc),
            )
        )

    def log_run_end(
        self,
        *,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        counts: Optional[Dict[str, int]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        self._append(
            self._make(
                event="run_end",
                inputs=inputs,
                outputs=outputs,
                counts=counts,
                metrics=metrics,
            )
        )

    # -----------------------------
    # Optional: reduce boilerplate
    # -----------------------------

    @contextmanager
    def step(
        self,
        *,
        step: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
    ) -> Iterator[None]:
        """
        Context manager for step logging:

          with rm.step(step="00_prepare", outputs=[...]):
              run_step_00(...)
        """
        self.log_step_start(step=step, inputs=inputs, outputs=outputs)
        try:
            yield
        except Exception as e:
            self.log_step_error(step=step, exc=e, inputs=inputs, outputs=outputs)
            raise
        else:
            self.log_step_end(step=step, inputs=inputs, outputs=outputs)


'''
#how to run:
from ccir.paths import Paths
from ccir.utils.run_metadata import RunMetadataLogger

paths = Paths(run_id="pilot1")
paths.ensure_shared_dirs()
paths.ensure_run_dirs()

rm = RunMetadataLogger(
    paths=paths,
    code_version="v0.1",      # required by Lineage
    config_hash=config_hash,  # optional
)

rm.log_run_start()

rm.log_step_start(step="00_prepare_euroverdict")
run_step_00(...)
rm.log_step_end(step="00_prepare_euroverdict", counts={"rows_written": 123})

rm.log_run_end()

#example:
rm.log_run_start(inputs=[str(paths.claims_all_jsonl)])

with rm.step(
    step="00_prepare_euroverdict",
    outputs=[str(paths.claims_all_jsonl)]
):
    run_step_00(paths)

with rm.step(
    step="01_generate_claim_subset",
    inputs=[str(paths.claims_all_jsonl)],
    outputs=[str(paths.claims_for_llms_jsonl)]
):
    run_step_01(paths)

rm.log_run_end()
'''