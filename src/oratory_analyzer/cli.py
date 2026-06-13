"""Command-line interface for the Oratory Analyzer.

Usage
-----
    oratory-analyzer analyze SPEECH.mp4 --out report/ [options]

Run ``oratory-analyzer analyze --help`` for the full option list.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import PipelineConfig
from .heartbeat import ProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oratory-analyzer",
        description="Analyze a speaker's video and report on oratory technique.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a video file.")
    analyze.add_argument("video", help="Path to the input video file.")
    analyze.add_argument(
        "-o", "--out", default="oratory_report", help="Output directory for the report."
    )
    analyze.add_argument(
        "--sample-fps", type=float, default=12.0,
        help="Frames per second to analyze (default: 12).",
    )
    analyze.add_argument(
        "--max-faces", type=int, default=3,
        help="Max faces detected per frame for speaker selection (default: 3).",
    )
    analyze.add_argument("--no-face", action="store_true", help="Skip face analysis.")
    analyze.add_argument("--no-pose", action="store_true", help="Skip pose analysis.")
    analyze.add_argument("--no-plots", action="store_true", help="Skip chart generation.")
    analyze.add_argument(
        "--annotated-video", action="store_true",
        help="Also render an annotated mp4 overlaying tracked landmarks.",
    )
    analyze.add_argument(
        "--quiet", action="store_true", help="Suppress progress output."
    )

    live = sub.add_parser(
        "live", help="Open the webcam and overlay tracked landmarks in real time."
    )
    live.add_argument(
        "-c", "--camera", type=int, default=0, help="Camera index (default: 0)."
    )
    live.add_argument("--width", type=int, default=None, help="Requested capture width.")
    live.add_argument("--height", type=int, default=None, help="Requested capture height.")
    live.add_argument("--no-face", action="store_true", help="Skip face mesh overlay.")
    live.add_argument("--no-pose", action="store_true", help="Skip pose skeleton overlay.")
    live.add_argument("--no-mirror", action="store_true", help="Do not mirror the preview.")
    live.add_argument("--no-hud", action="store_true", help="Hide the status HUD.")
    return parser


def _run_analyze(args: argparse.Namespace) -> int:
    if not os.path.exists(args.video):
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    config = PipelineConfig(
        sample_fps=args.sample_fps,
        max_num_faces=args.max_faces,
        analyze_face=not args.no_face,
        analyze_pose=not args.no_pose,
        write_annotated_video=args.annotated_video,
        output_dir=args.out,
        with_plots=not args.no_plots,
    )

    progress = None if args.quiet else ProgressReporter(
        heartbeat_path=os.path.join(args.out, "run_heartbeat.log")
    )
    if progress:
        os.makedirs(args.out, exist_ok=True)

    # Import here so `--help` works even without MediaPipe installed.
    from .pipeline import OratoryAnalysisPipeline

    pipeline = OratoryAnalysisPipeline.with_mediapipe(config=config, progress=progress)
    try:
        written = pipeline.run(args.video)
    finally:
        pipeline.close()

    print("\nReport artifacts:")
    for kind, path in written.items():
        print(f"  {kind:18s} {path}")
    print(f"\nOpen {written.get('html', '')} in a browser for the full report.")
    return 0


def _run_live(args: argparse.Namespace) -> int:
    if args.no_face and args.no_pose:
        print("error: cannot disable both face and pose overlays", file=sys.stderr)
        return 2

    from .live import LiveTracker

    tracker = LiveTracker.with_mediapipe(
        analyze_face=not args.no_face,
        analyze_pose=not args.no_pose,
        mirror=not args.no_mirror,
        show_hud=not args.no_hud,
    )
    print("Opening camera… look at yourself; press 'q' or Esc in the window to quit.")
    try:
        tracker.run(camera_index=args.camera, width=args.width, height=args.height)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "live":
        return _run_live(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
