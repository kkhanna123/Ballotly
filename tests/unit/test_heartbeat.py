"""Tests for the progress/heartbeat reporter."""

from __future__ import annotations

import io

from oratory_analyzer.heartbeat import NullProgress, ProgressReporter


def test_logs_to_stream_with_elapsed():
    stream = io.StringIO()
    clock = iter([100.0, 102.5])  # start (in __init__), then one log call
    reporter = ProgressReporter(stream=stream, clock=lambda: next(clock))
    reporter.log("hello")
    out = stream.getvalue()
    assert "hello" in out
    assert "+   2.5s" in out


def test_appends_to_heartbeat_file(tmp_path):
    hb = tmp_path / "hb.log"
    stream = io.StringIO()
    reporter = ProgressReporter(stream=stream, heartbeat_path=str(hb))
    reporter.log("first")
    reporter.log("second")
    contents = hb.read_text()
    assert "first" in contents and "second" in contents


def test_null_progress_is_silent():
    # Should not raise and should produce no output side effects.
    NullProgress().log("ignored")
