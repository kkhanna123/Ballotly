# 🫀 Build Heartbeat — Oratory Analyzer

This file is updated continuously while you're away so you can see exactly what
was built, what decisions were made, and where things stand.

**Project:** Face & posture tracking software that ingests a video of a speaker,
auto-segments to identify the primary speaker, tracks facial + posture landmarks,
and generates a report on how to improve oratory ability for debate competition.

---

## Key decisions / pivots

- **2026-06-13 — Engine: MediaPipe (pivot away from DeepLabCut).**
  DeepLabCut needs hand-labeled training data + a GPU to train a model — not
  viable for an unattended, run-immediately tool. MediaPipe ships *pretrained*
  Face Mesh (468 landmarks) and Pose (33 landmarks) models that run on CPU with
  zero training. Landmark extraction is hidden behind an abstract
  `LandmarkExtractor` interface, so DeepLabCut / Roboflow `supervision` / MMPose
  can be swapped in later without touching metrics, analysis, or reporting.
- **2026-06-13 — Python 3.12 (x86_64) venv.** System default was Python 3.8.5
  (too old for MediaPipe). Used `/usr/local/bin/python3.12`.
- **2026-06-13 — Architecture: pure-Python core, thin I/O shell.** Domain models
  (landmarks as dataclasses) are the contract. Everything analytical (metrics,
  scoring, report) is pure Python operating on those dataclasses, so ~all logic
  is unit-testable with synthetic data — no video files or ML models needed in
  the test suite. MediaPipe/OpenCV live only at the edges.

---

## Status log

- `2026-06-13 10:50` — Environment probed; venv created (py3.12); dependency
  install (mediapipe/opencv/numpy/jinja2/matplotlib/pytest) kicked off in
  background. Project skeleton + task list created. Starting domain layer.
- `2026-06-13 11:10` — Deps installed & verified (mediapipe 0.10.14, cv2 4.11,
  numpy 1.26.4). Domain layer done (Point3D, Face/Pose/FrameLandmarks,
  BoundingBox w/ IoU, geometry: angles, tilt, lean, EAR, series stats) — 55
  tests green. Landmark layer done: abstract Face/Pose extractors,
  LandmarkPipeline, MediaPipe adapters, scripted fakes — 7 tests green.
- `2026-06-13 11:25` — Metrics layer done: eye_contact, head_stability,
  posture, gestures, facial_expressivity + registry. Scoring helpers
  (fraction/band/tolerance). All metrics output a 0..100 "higher=better" score
  with coverage + per-frame series for plots. 35 metric tests green
  (good-clip > bad-clip behaviour asserted for each). Total: 97 tests green.
  Now building analysis/scoring (aggregate metrics → overall grade +
  recommendations).
- `2026-06-13 11:35` — Analysis/scoring done: weighted overall grade (A–F),
  strengths/weaknesses, prioritized recommendations with practice drills.
  Report layer done: JSON + Markdown + HTML (Jinja2, styled, CSS score bars)
  + matplotlib charts, all behind a renderer interface with a ReportBuilder.
- `2026-06-13 11:50` — Detection done: pure IoU tracker + primary-speaker
  selector (presence × size × centrality), engine-agnostic & fully tested.
  Video I/O done: OpenCV reader (with fps subsampling) / writer / annotator.
  Pipeline orchestrator + argparse CLI done. 157 tests green.
- `2026-06-13 12:05` — **END-TO-END VALIDATED ON REAL FOOTAGE.** Downloaded a
  real talking-head clip; full MediaPipe run: detected 3 face tracks, correctly
  selected the speaker (present 84–100% of frames), scored 67.7/100 (grade D),
  wrote JSON/MD/HTML + charts + annotated mp4. Visually confirmed the annotated
  overlay tracks the right face + pose skeleton.
- `2026-06-13 12:10` — Real data exposed a bug: a head-and-shoulders shot has no
  visible hands, so the gesture metric had 0% coverage but was still flagged as
  a weakness + generated advice. **Fixed:** metrics below `min_assess_coverage`
  are now reported as "not assessed" — excluded from grading, strengths/
  weaknesses, and recommendations (we don't coach on what we couldn't measure).
- `2026-06-13 12:20` — README written; added integration tests driving the full
  `run()` over a synthetic video (covers decode → annotate → report). Final:
  **164 tests green, ~91% coverage** (analytical core 84–100%). BUILD COMPLETE.

---

## ✅ Final status: COMPLETE

- `./venv/bin/python -m pytest` → 164 passed, ~91% coverage.
- Example output committed under `samples/report/` (json/md/html + charts +
  annotated.mp4), generated from `samples/face_clip.mp4`.
- Run it yourself:
  `./venv/bin/python -m oratory_analyzer.cli analyze <video> --out report --annotated-video`

**Pivots made along the way (all reasonable, documented above):**
1. DeepLabCut → MediaPipe (no training/GPU needed; runs immediately).
2. System Python 3.8 → 3.12 venv (MediaPipe requirement).
3. Added explicit "not assessed" handling after real footage revealed the
   zero-coverage gesture bug.

- `2026-06-13 (follow-up 2)` — Added **dedicated hand + body tracking**. Body was
  already covered by Pose (33 pts); added MediaPipe **Hands** (21 pts/hand, up to
  2 hands) behind a new `HandExtractor` interface. New `HandLandmarks` domain
  model + `Hands` indices; `FrameLandmarks.hands`. New `HandGestureMetric`
  (finger-level openness + hand motion in hand-widths/sec) with a `HANDS` metric
  category. Wired through pipeline, annotator (magenta hand skeleton), live HUD
  (hand count), config + CLI (`--no-hands`). Verified end-to-end on a classroom
  clip: hands detected (42% coverage), scored, and rendered in the annotated
  video (confirmed the magenta hand overlay visually). **195 tests green.**
- `2026-06-13 (follow-up)` — Added **live webcam mode**: `live.py` (`LiveTracker`)
  opens the laptop camera and overlays the face mesh + pose skeleton in real time
  with a HUD (FPS, detection status, live eye-contact + posture cues). Wired as
  `oratory-analyzer live` CLI subcommand. Per-frame logic isolated in
  `process_frame()` and unit-tested with fakes; verified headlessly through real
  MediaPipe. Wrote `HOW_TO_RUN.md` (setup, live + analyze modes, macOS camera
  permission, troubleshooting). **184 tests green.**

---

## Architecture (target)

```
src/oratory_analyzer/
  domain/      landmarks, geometry, frame  (pure data + math)
  landmarks/   LandmarkExtractor ABC + MediaPipe face/pose impls + fakes
  detection/   speaker selection + IoU tracker (identity persistence)
  video/       OpenCV reader/writer + annotation
  metrics/     eye contact, head stability, posture, gestures, expressivity
  analysis/    aggregation + scoring + recommendations
  report/      JSON / Markdown / HTML renderers + plots
  pipeline.py  orchestrator
  cli.py       entrypoint
tests/         unit (pure, fast) + integration (pipeline w/ fakes)
```

## How to run (once built)

```bash
./venv/bin/python -m oratory_analyzer.cli analyze path/to/speech.mp4 --out report/
# or, after `pip install -e .`:
oratory-analyzer analyze speech.mp4 --out report/
```
