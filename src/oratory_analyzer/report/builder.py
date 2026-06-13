"""Report builder: writes JSON + Markdown + HTML (and plots) to a directory."""

from __future__ import annotations

import os
from typing import Dict, List

from ..analysis.aggregator import OverallAssessment
from .html import HtmlReportRenderer
from .json_report import JsonReportRenderer
from .markdown import MarkdownReportRenderer
from .plots import PlotGenerator


class ReportBuilder:
    """Generates the full report bundle for an assessment.

    Writes ``report.json``, ``report.md`` and ``report.html`` plus a ``figures/``
    directory of PNG charts. Returns the mapping of artifact name → path.
    """

    def __init__(
        self,
        *,
        basename: str = "report",
        with_plots: bool = True,
    ) -> None:
        self.basename = basename
        self.with_plots = with_plots

    def build(self, assessment: OverallAssessment, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        written: Dict[str, str] = {}

        figures: Dict[str, str] = {}
        if self.with_plots:
            try:
                figures = PlotGenerator().generate(assessment, output_dir)
                written.update(
                    {f"figure:{k}": os.path.join(output_dir, v) for k, v in figures.items()}
                )
            except Exception as exc:  # plotting must never sink the whole report
                assessment.notes.append(f"Chart generation skipped: {exc}")

        renderers = [
            JsonReportRenderer(),
            MarkdownReportRenderer(),
            HtmlReportRenderer(figures=figures),
        ]
        for renderer in renderers:
            content = renderer.render(assessment)
            path = os.path.join(output_dir, f"{self.basename}.{renderer.extension}")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            written[renderer.extension] = path

        return written
