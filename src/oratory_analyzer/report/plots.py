"""Timeline / summary plot generation (matplotlib, headless).

Kept separate from the text renderers so the renderers stay pure and
dependency-light. This module is the only one that imports matplotlib and is
exercised via the report integration test rather than the pure unit suite.
"""

from __future__ import annotations

import math
import os
from typing import Dict

from ..analysis.aggregator import OverallAssessment


def _use_agg():
    import matplotlib

    matplotlib.use("Agg")  # headless, no display needed
    import matplotlib.pyplot as plt

    return plt


class PlotGenerator:
    """Writes summary PNGs into ``figures_dir`` and returns their file names."""

    def __init__(self, figures_dirname: str = "figures") -> None:
        self.figures_dirname = figures_dirname

    def generate(self, assessment: OverallAssessment, output_dir: str) -> Dict[str, str]:
        plt = _use_agg()
        fig_dir = os.path.join(output_dir, self.figures_dirname)
        os.makedirs(fig_dir, exist_ok=True)
        produced: Dict[str, str] = {}

        # 1) Overall metric bar chart.
        names = [r.title for r in assessment.metrics.values()]
        scores = [r.score for r in assessment.metrics.values()]
        if names:
            fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1))
            colors = [
                "#2e7d32" if s >= 80 else "#f9a825" if s >= 60 else "#c62828"
                for s in scores
            ]
            ax.barh(names, scores, color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.set_title(f"Oratory metrics (overall {assessment.overall_score}/100)")
            ax.invert_yaxis()
            for i, s in enumerate(scores):
                ax.text(min(s + 1, 95), i, f"{s:.0f}", va="center", fontsize=9)
            fig.tight_layout()
            rel = os.path.join(self.figures_dirname, "metric_scores.png")
            fig.savefig(os.path.join(output_dir, rel), dpi=120, bbox_inches="tight")
            plt.close(fig)
            produced["metric_scores"] = rel

        # 2) Per-metric timelines (signals over frame index), skipping NaNs.
        series_metrics = {
            name: r for name, r in assessment.metrics.items() if r.series
        }
        if series_metrics:
            n = len(series_metrics)
            fig, axes = plt.subplots(n, 1, figsize=(8, 1.8 * n), squeeze=False)
            for ax, (name, r) in zip(axes[:, 0], series_metrics.items()):
                ys = [v for v in r.series if not math.isnan(v)]
                xs = [i for i, v in enumerate(r.series) if not math.isnan(v)]
                ax.plot(xs, ys, color="#1565c0", linewidth=1.0)
                ax.set_title(f"{r.title} (signal)", fontsize=9, loc="left")
                ax.set_ylabel("value", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[-1, 0].set_xlabel("frame")
            fig.tight_layout()
            rel = os.path.join(self.figures_dirname, "timelines.png")
            fig.savefig(os.path.join(output_dir, rel), dpi=120, bbox_inches="tight")
            plt.close(fig)
            produced["timelines"] = rel

        return produced
