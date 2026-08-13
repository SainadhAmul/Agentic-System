"""Logging utility for Agent7.

Captures all stdout and stderr during an agent run and mirrors them simultaneously
to the console and to log files:
  - Per-run log: `logs/run_<run_id>_<timestamp>.log`
  - Cumulative log: `logs/agent7.log`
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

LOGS_DIR = Path(__file__).parent / "logs"


class _TeeStream:
    """A stream wrapper that writes to multiple underlying streams simultaneously."""

    def __init__(self, *streams: TextIO):
        self.streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

    def reconfigure(self, **kwargs) -> None:
        for s in self.streams:
            if hasattr(s, "reconfigure"):
                try:
                    s.reconfigure(**kwargs)
                except Exception:
                    pass


class RunLogger:
    """Context manager to capture all prints and errors for a specific agent run."""

    def __init__(self, run_id: str, query: str = ""):
        self.run_id = run_id
        self.query = query
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = LOGS_DIR / f"run_{run_id}_{self.timestamp}.log"
        self.cumulative_log_path = LOGS_DIR / "agent7.log"
        self._run_file: TextIO | None = None
        self._cumulative_file: TextIO | None = None
        self._orig_stdout: TextIO | None = None
        self._orig_stderr: TextIO | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> RunLogger:
        self._start_time = time.time()
        self._run_file = open(self.run_log_path, "a", encoding="utf-8")
        self._cumulative_file = open(self.cumulative_log_path, "a", encoding="utf-8")

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        sys.stdout = _TeeStream(self._orig_stdout, self._run_file, self._cumulative_file)
        sys.stderr = _TeeStream(self._orig_stderr, self._run_file, self._cumulative_file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = time.time() - self._start_time
        if exc_val is not None:
            import traceback
            sys.stderr.write(f"\n[logger] Run failed with error after {duration:.2f}s:\n")
            traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stderr)
        else:
            sys.stdout.write(f"\n[logger] Run completed in {duration:.2f}s\n")
            sys.stdout.write(f"[logger] Log saved to: {self.run_log_path.resolve()}\n")

        # Restore original streams
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        if self._run_file:
            try:
                self._run_file.close()
            except Exception:
                pass
        if self._cumulative_file:
            try:
                self._cumulative_file.close()
            except Exception:
                pass
