"""JSON report renderer (machine-readable, for dashboards or further analysis)."""

from __future__ import annotations

import json

from ..analysis.aggregator import OverallAssessment
from .base import ReportRenderer


class JsonReportRenderer(ReportRenderer):
    extension = "json"

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def render(self, assessment: OverallAssessment) -> str:
        return json.dumps(assessment.to_dict(), indent=self.indent, sort_keys=False)
