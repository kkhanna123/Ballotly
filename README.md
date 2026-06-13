# Oratory Analyzer

Face & posture tracking that turns a video of a speaker into a coaching report
on their **debate / oratory delivery**. It auto-segments the footage to identify
the **primary speaker**, tracks facial and body landmarks frame-by-frame, scores
five delivery dimensions, and writes a JSON / Markdown / HTML report (plus an
optional annotated video).

```
video ──► landmark extraction ──► speaker selection ──► metrics ──► scoring ──► report
          (MediaPipe Face Mesh        (IoU tracking +     (eye contact,   (weighted   (JSON/MD/HTML
           + Pose + Hands,             dominance score)    posture,        grade)      + plots + mp4)
           pluggable)                                      gestures, …)
```

## What it measures

| Dimension | Landmarks | What it rewards |
|---|---|---|
| **Eye contact** | Face mesh + irises | Facing the audience; not buried in notes |
| **Head stability** | Face mesh | Composed head carriage vs. nervous swaying |
| **Facial expressivity** | Face mesh | Animated brow/mouth vs. flat affect |
| **Posture & stance** | Pose (body) | Upright, level shoulders, no slouch |
| **Hand gestures** | Pose (body) | Broad arm/wrist gesturing vs. frozen or frantic |
| **Hand gestures (detailed)** | Hands | Finger-level gesturing + articulation (21-pt hand model) |

Three tracking modalities run together: **face** (468-pt mesh + irises),
**body** (33-pt pose), and **hands** (21-pt hand model per hand, up to two).
Each can be toggled independently.

Every metric is normalized to **0–100 (higher is better)**, combined into a
weighted overall grade (A–F), and turned into prioritized, actionable
recommendations with practice drills. Metrics whose landmarks aren't visible
(e.g. hands off-screen in a head-and-shoulders shot) are reported as
*not assessed* rather than penalized.

## Install

Requires Python 3.9–3.12 (MediaPipe constraint).

```bash
python3.12 -m venv venv
./venv/bin/python -m pip install -e .
```

## Usage

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for a step-by-step guide (setup, live mode,
analyze mode, camera permissions, troubleshooting).

### Live mode — watch the lines on your face in real time

```bash
# Open the laptop camera with face-mesh + pose overlays and a live cue HUD
./venv/bin/python -m oratory_analyzer.cli live          # press q / Esc to quit
./venv/bin/python -m oratory_analyzer.cli live --no-pose # face mesh only (lighter)
```

### Analyze mode

```bash
# Analyze a clip and write the report bundle to ./report
./venv/bin/python -m oratory_analyzer.cli analyze speech.mp4 --out report

# Faster scan, also render an annotated video overlaying the tracked landmarks
./venv/bin/python -m oratory_analyzer.cli analyze speech.mp4 --out report \
    --sample-fps 6 --annotated-video

# After `pip install -e .` the console script is available too:
oratory-analyzer analyze speech.mp4 --out report
```

Open `report/report.html` for the full styled report. `report/report.json` is
machine-readable for dashboards.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | `oratory_report` | Output directory |
| `--sample-fps N` | `12` | Frames per second to analyze (lower = faster) |
| `--max-faces N` | `3` | Faces detected per frame for speaker selection |
| `--no-face` / `--no-pose` | off | Skip a landmark group |
| `--no-plots` | off | Skip chart generation |
| `--annotated-video` | off | Render `annotated.mp4` |
| `--quiet` | off | Suppress progress output |

> **Footage tips:** a full-body or waist-up framing lets the posture and gesture
> metrics work; a tight head shot will analyze face metrics only. The speaker
> should be the most prominent/central person for reliable auto-selection.

## Architecture

A pure-Python analytical core wrapped by a thin I/O shell, so ~all logic is
unit-testable without video files or ML models.

```
src/oratory_analyzer/
  domain/      value objects: Point3D, Face/Pose/Hand/FrameLandmarks, BoundingBox, geometry
  landmarks/   Face/Pose/Hand extractor ABCs + MediaPipe adapters + scripted fakes
  detection/   IoU tracker + primary-speaker selector (engine-agnostic, pure)
  video/       OpenCV reader/writer + frame annotator (I/O edge)
  metrics/     the five oratory metrics + scoring helpers + registry
  analysis/    weighted aggregation, A–F grading, recommendation engine
  report/      JSON / Markdown / HTML renderers + matplotlib plots + builder
  pipeline.py  orchestrates the full flow
  cli.py       argparse entrypoint
```

**Swappable engine.** Landmark extraction sits behind `FaceExtractor` /
`PoseExtractor` / `HandExtractor` interfaces. MediaPipe is the default
(pretrained, CPU, no training required); DeepLabCut, Roboflow `supervision`, or
MMPose could be dropped in by implementing the same interface — nothing
downstream changes.

## Testing

```bash
./venv/bin/python -m pytest          # full suite
./venv/bin/python -m pytest --cov    # with coverage
./venv/bin/python -m pytest -m "not integration"   # fast pure-unit subset
```

164 tests; ~91% coverage. The pure core (domain, metrics, detection, analysis,
report) is exercised deterministically with synthetic landmarks via scripted
fake extractors; the MediaPipe/OpenCV edges are covered by integration tests and
the end-to-end run.

## Design notes & limitations

- Metrics are **heuristics** calibrated with sensible defaults (all thresholds
  are constructor arguments and easily tuned). They are coaching aids, not a
  substitute for a human judge.
- "Eye contact" infers gaze from head yaw + iris position — it cannot know where
  the audience actually is, so it treats *facing the camera* as the target.
- Audio (pace, filler words, volume) is **out of scope** for this version; it is
  a natural next module behind a similar interface.
