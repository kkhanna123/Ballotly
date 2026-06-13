"""Tests for the CLI argument parsing and dispatch (no real video)."""

from __future__ import annotations

import pytest

from oratory_analyzer.cli import build_parser, main


def test_parser_requires_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_analyze_defaults():
    args = build_parser().parse_args(["analyze", "clip.mp4"])
    assert args.video == "clip.mp4"
    assert args.out == "oratory_report"
    assert args.sample_fps == 12.0
    assert args.max_faces == 3
    assert not args.no_face and not args.no_pose


def test_analyze_flags():
    args = build_parser().parse_args(
        ["analyze", "clip.mp4", "-o", "out", "--sample-fps", "5",
         "--no-pose", "--no-hands", "--annotated-video", "--quiet"]
    )
    assert args.out == "out"
    assert args.sample_fps == 5.0
    assert args.no_pose and args.no_hands and not args.no_face
    assert args.annotated_video and args.quiet


def test_main_missing_video_returns_error_code():
    # nonexistent file => exit code 2, no MediaPipe import triggered
    assert main(["analyze", "does_not_exist_12345.mp4"]) == 2


def test_live_defaults():
    args = build_parser().parse_args(["live"])
    assert args.command == "live"
    assert args.camera == 0
    assert not args.no_face and not args.no_pose and not args.no_mirror


def test_live_flags():
    args = build_parser().parse_args(
        ["live", "-c", "1", "--width", "1280", "--no-pose", "--no-hud", "--no-mirror"]
    )
    assert args.camera == 1 and args.width == 1280
    assert args.no_pose and args.no_hud and args.no_mirror


def test_live_all_overlays_disabled_errors():
    # disabling face, pose, and hands => error code 2, no camera/MediaPipe touched
    assert main(["live", "--no-face", "--no-pose", "--no-hands"]) == 2


def test_live_disabling_two_is_allowed_at_parse():
    args = build_parser().parse_args(["live", "--no-face", "--no-pose"])
    assert args.no_face and args.no_pose and not args.no_hands
