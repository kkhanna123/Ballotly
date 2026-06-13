"""Oratory Analyzer — face & posture tracking for debate/oratory coaching.

The package is organized as a pure-Python analytical core wrapped by a thin
I/O shell:

* :mod:`oratory_analyzer.domain` — value objects (landmarks, geometry, frames).
* :mod:`oratory_analyzer.landmarks` — pluggable landmark extractors (MediaPipe).
* :mod:`oratory_analyzer.detection` — primary-speaker selection + tracking.
* :mod:`oratory_analyzer.video` — OpenCV video read/write + annotation.
* :mod:`oratory_analyzer.metrics` — oratory metrics computed per frame.
* :mod:`oratory_analyzer.analysis` — aggregation, scoring, recommendations.
* :mod:`oratory_analyzer.report` — JSON / Markdown / HTML report rendering.
* :mod:`oratory_analyzer.pipeline` — orchestrates all stages end to end.
"""

__version__ = "0.1.0"
