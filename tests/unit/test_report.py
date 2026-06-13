"""Tests for report renderers and the builder."""

from __future__ import annotations

import json
import os

import pytest

from oratory_analyzer.analysis import Analyzer
from oratory_analyzer.metrics.base import FACE, POSE, MetricResult
from oratory_analyzer.report import (
    HtmlReportRenderer,
    JsonReportRenderer,
    MarkdownReportRenderer,
    ReportBuilder,
)


def _assessment():
    results = {
        "eye_contact": MetricResult(
            name="eye_contact", score=40, summary="Looked away a lot.",
            category=FACE, title="Eye Contact", coverage=0.9,
            series=[0.1, 0.2, float("nan"), 0.15],
        ),
        "posture": MetricResult(
            name="posture", score=88, summary="Upright.", category=POSE,
            title="Posture & Stance", coverage=1.0, series=[80.0, 90.0, 95.0],
        ),
    }
    return Analyzer().analyze(results, frames_analyzed=4, frames_with_speaker=4)


class TestJsonRenderer:
    def test_valid_json(self):
        out = JsonReportRenderer().render(_assessment())
        data = json.loads(out)
        assert data["grade"] in {"A", "B", "C", "D", "F"}
        assert "eye_contact" in data["metrics"]
        assert data["metrics"]["eye_contact"]["title"] == "Eye Contact"


class TestMarkdownRenderer:
    def test_contains_key_sections(self):
        out = MarkdownReportRenderer().render(_assessment())
        assert "# Oratory Analysis Report" in out
        assert "## Metric breakdown" in out
        assert "## Recommendations" in out
        # low-scoring eye_contact should produce a recommendation heading
        assert "eye" in out.lower()

    def test_extension(self):
        assert MarkdownReportRenderer().extension == "md"


class TestHtmlRenderer:
    def test_well_formed_html(self):
        out = HtmlReportRenderer().render(_assessment())
        assert out.startswith("<!DOCTYPE html>")
        assert "Oratory Analysis Report" in out
        assert "Eye Contact" in out
        assert "Posture &amp; Stance" in out or "Posture & Stance" in out

    def test_includes_figures_when_present(self):
        out = HtmlReportRenderer(figures={"metric_scores": "figures/metric_scores.png"}).render(
            _assessment()
        )
        assert "figures/metric_scores.png" in out


@pytest.mark.integration
class TestReportBuilder:
    def test_writes_all_artifacts(self, tmp_path):
        written = ReportBuilder().build(_assessment(), str(tmp_path))
        assert os.path.exists(written["json"])
        assert os.path.exists(written["md"])
        assert os.path.exists(written["html"])
        # JSON must be parseable
        with open(written["json"]) as fh:
            json.load(fh)

    def test_generates_plots(self, tmp_path):
        ReportBuilder(with_plots=True).build(_assessment(), str(tmp_path))
        fig_dir = tmp_path / "figures"
        assert fig_dir.exists()
        pngs = list(fig_dir.glob("*.png"))
        assert len(pngs) >= 1

    def test_without_plots(self, tmp_path):
        ReportBuilder(with_plots=False).build(_assessment(), str(tmp_path))
        assert not (tmp_path / "figures").exists()
