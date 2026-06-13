"""Progress reporting for long-running analysis.

``ProgressReporter`` prints timestamped progress to a stream and can optionally
append to a heartbeat file, so a long unattended run leaves a trail you can tail.
``NullProgress`` is the silent default used in tests.
"""

from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


class ProgressReporter:
    """Logs timestamped progress lines to a stream and/or a heartbeat file."""

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        heartbeat_path: Optional[str] = None,
        *,
        clock=time.time,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.heartbeat_path = heartbeat_path
        self._clock = clock
        self._start = clock()

    def log(self, message: str) -> None:
        elapsed = self._clock() - self._start
        line = f"[+{elapsed:6.1f}s] {message}"
        print(line, file=self.stream, flush=True)
        if self.heartbeat_path:
            try:
                with open(self.heartbeat_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # never let logging crash the run


class NullProgress(ProgressReporter):
    """A no-op reporter (used by default and in tests)."""

    def __init__(self) -> None:  # noqa: D107 - intentionally minimal
        self.heartbeat_path = None

    def log(self, message: str) -> None:  # pragma: no cover - trivial
        pass
