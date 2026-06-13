# How to Run — Oratory Analyzer

Two ways to use the tool:

1. **Live mode** — open your laptop camera and watch the face-mesh + pose lines
   drawn on you in real time (with live gaze/posture cues).
2. **Analyze mode** — process a recorded video into a full coaching report.

---

## 1. One-time setup

You need **Python 3.9–3.12** (MediaPipe doesn't support 3.13+ yet). This repo was
built and tested with `python3.12`.

```bash
cd /Users/nitinkhanna/Desktop/Ballotly/FaceAndPostureTracking

# Create the virtual environment (only once)
python3.12 -m venv venv

# Install the package and its dependencies into the venv
./venv/bin/python -m pip install -e .
```

That's it. Every command below calls the venv's Python directly
(`./venv/bin/python`), so you don't need to "activate" anything. If you prefer,
you can `source venv/bin/activate` and then use `oratory-analyzer …` directly.

Verify it installed:

```bash
./venv/bin/python -m oratory_analyzer.cli --help
```

---

## 2. Live camera mode (see the lines on your face)

```bash
./venv/bin/python -m oratory_analyzer.cli live
```

A window opens showing your mirrored webcam feed with:

- a green **face mesh** (468 points) drawn on your face,
- iris points and a yellow **speaker** box,
- an orange **pose skeleton** across your shoulders/arms/torso,
- a HUD (top-left) with **FPS**, detection status, and live **eye-contact** and
  **posture** cues that update as you move.

**Press `q` or `Esc` (with the window focused) to quit.**

### Useful flags

| Command | Effect |
|---|---|
| `live -c 1` | Use camera index 1 (external/second webcam) |
| `live --width 1280 --height 720` | Request a capture resolution |
| `live --no-pose` | Face mesh only (faster, good for a head shot) |
| `live --no-face` | Pose skeleton only |
| `live --no-mirror` | Don't mirror the preview |
| `live --no-hud` | Hide the status overlay |

### macOS camera permission (important)

The first time you run `live`, macOS will ask the **terminal app you launched it
from** (Terminal.app, iTerm, VS Code, etc.) for camera access. If you don't see a
prompt and instead get a "Could not open camera" error:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable your terminal app.
3. Quit and reopen the terminal, then run `live` again.

> Tip: inside this Claude Code session you can run it yourself by typing
> `! ./venv/bin/python -m oratory_analyzer.cli live` at the prompt — the camera
> window will open on your machine.

---

## 3. Analyze a recorded video (full report)

```bash
./venv/bin/python -m oratory_analyzer.cli analyze path/to/speech.mp4 --out report
```

Outputs land in `report/`:

| File | What it is |
|---|---|
| `report.html` | Styled, human-readable report — **open this in a browser** |
| `report.md` | Same report in Markdown |
| `report.json` | Machine-readable scores/stats |
| `figures/*.png` | Score bar chart + per-metric timelines |
| `run_heartbeat.log` | Timestamped progress log |

### Useful flags

| Command | Effect |
|---|---|
| `--sample-fps 6` | Analyze ~6 frames/sec (faster on long clips) |
| `--annotated-video` | Also write `report/annotated.mp4` with the overlays |
| `--no-pose` / `--no-face` | Skip a landmark group |
| `--no-plots` | Skip chart generation |
| `--quiet` | Suppress progress output |

**Footage tips:** a waist-up or full-body framing lets posture *and* gesture
metrics work. A tight head shot analyzes face metrics only (gestures will show as
*not assessed*). The speaker should be the most prominent/central person so the
auto speaker-selection locks onto the right face.

### Try it on a sample clip

Any video of a speaker works. The repo doesn't bundle a clip (videos are large
and not committed), but you can grab a public talking-head sample to test with:

```bash
mkdir -p samples
curl -fsSL -o samples/face_clip.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4

./venv/bin/python -m oratory_analyzer.cli analyze samples/face_clip.mp4 \
    --out samples/report --sample-fps 6 --annotated-video
open samples/report/report.html        # macOS
```

---

## 4. Running the tests

```bash
./venv/bin/python -m pytest                      # full suite (≈180 tests)
./venv/bin/python -m pytest -m "not integration" # fast, pure-unit subset
./venv/bin/python -m pytest --cov                # with coverage
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not open camera` | Grant camera permission (see §2); close other apps using the camera; try `-c 1`. |
| Live window opens but is black / no lines | Make sure your face is lit and in frame; MediaPipe needs a real face (it won't track drawings/avatars). |
| `ModuleNotFoundError: oratory_analyzer` | Run via `./venv/bin/python …`, or `pip install -e .` inside the venv. |
| Slow / low FPS in live mode | Use `--no-pose` (face-only is lighter) or request a smaller resolution. |
| `analyze` says no metrics could be computed | No face/pose was detected — check the speaker is visible and well-lit. |
| MediaPipe install fails | Confirm Python is 3.9–3.12: `./venv/bin/python --version`. |
