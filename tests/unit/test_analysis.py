"""Tests for scoring, recommendations, and aggregation."""

from __future__ import annotations

import pytest

from oratory_analyzer.analysis import (
    Analyzer,
    RecommendationEngine,
    score_to_grade,
    score_to_label,
)
from oratory_analyzer.domain.frame import VideoMetadata
from oratory_analyzer.metrics.base import FACE, POSE, MetricResult


def _result(name, score, *, coverage=1.0, weight=1.0, category=FACE):
    return MetricResult(
        name=name, score=score, summary="", category=category,
        coverage=coverage, weight=weight,
    )


class TestGradeMapping:
    @pytest.mark.parametrize(
        "score,grade",
        [(95, "A"), (90, "A"), (85, "B"), (75, "C"), (65, "D"), (50, "F"), (0, "F")],
    )
    def test_score_to_grade(self, score, grade):
        assert score_to_grade(score) == grade

    def test_labels(self):
        assert score_to_label(95) == "Excellent"
        assert score_to_label(50) == "Needs work"


class TestRecommendationEngine:
    def test_only_low_metrics_generate_recs(self):
        results = {
            "eye_contact": _result("eye_contact", 95),
            "posture": _result("posture", 40, category=POSE),
        }
        recs = RecommendationEngine(focus_threshold=70).generate(results)
        assert len(recs) == 1
        assert recs[0].metric_name == "posture"

    def test_recs_sorted_by_priority(self):
        results = {
            # weight default 1 here; lower score => higher priority
            "eye_contact": _result("eye_contact", 65, weight=1.5),
            "gestures": _result("gestures", 65, weight=1.0, category=POSE),
        }
        recs = RecommendationEngine(focus_threshold=70).generate(results)
        assert [r.metric_name for r in recs] == ["eye_contact", "gestures"]

    def test_rec_has_actionable_drill(self):
        results = {"eye_contact": _result("eye_contact", 30)}
        recs = RecommendationEngine().generate(results)
        assert recs[0].drill  # non-empty actionable text
        assert recs[0].to_dict()["metric"] == "eye_contact"


class TestAnalyzer:
    def test_weighted_overall_score(self):
        results = {
            "a": _result("a", 100, weight=3.0),
            "b": _result("b", 0, weight=1.0),
        }
        assessment = Analyzer().analyze(results)
        # (100*3 + 0*1) / (3+1) = 75
        assert assessment.overall_score == pytest.approx(75.0)
        assert assessment.grade == "C"

    def test_coverage_downweights_metric(self):
        # 'b' has full score but low coverage => still assessed but barely counts
        results = {
            "a": _result("a", 90, weight=1.0, coverage=1.0),
            "b": _result("b", 0, weight=1.0, coverage=0.2),
        }
        assessment = Analyzer().analyze(results)
        # naive mean would be 45; coverage-weighting keeps it well above that.
        assert assessment.overall_score > 70

    def test_near_zero_coverage_metric_marked_unavailable(self):
        results = {
            "a": _result("a", 90, coverage=1.0),
            "gestures": _result("gestures", 0, coverage=0.0),
        }
        assessment = Analyzer().analyze(results)
        assert "gestures" in assessment.unavailable
        assert "gestures" not in assessment.metrics
        assert "gestures" not in assessment.weaknesses
        # unavailable metrics must not generate recommendations
        assert all(r.metric_name != "gestures" for r in assessment.recommendations)
        assert assessment.overall_score == pytest.approx(90.0)

    def test_all_unavailable_raises(self):
        results = {"a": _result("a", 50, coverage=0.0)}
        with pytest.raises(ValueError, match="enough visible frames"):
            Analyzer().analyze(results)

    def test_strengths_and_weaknesses(self):
        results = {
            "a": _result("a", 90),
            "b": _result("b", 50),
            "c": _result("c", 70),
        }
        assessment = Analyzer().analyze(results)
        assert "a" in assessment.strengths
        assert "b" in assessment.weaknesses
        assert "c" not in assessment.strengths and "c" not in assessment.weaknesses

    def test_low_coverage_note_added(self):
        results = {"a": _result("a", 80, coverage=0.2)}
        assessment = Analyzer().analyze(results)
        assert any("low confidence" in n for n in assessment.notes)

    def test_duration_from_video_meta(self):
        results = {"a": _result("a", 80)}
        meta = VideoMetadata(width=640, height=480, fps=30, frame_count=900)
        assessment = Analyzer().analyze(
            results, frames_analyzed=900, frames_with_speaker=850, video_meta=meta
        )
        assert assessment.duration_seconds == pytest.approx(30.0)
        assert assessment.tracking_coverage == pytest.approx(850 / 900)

    def test_empty_results_raise(self):
        with pytest.raises(ValueError):
            Analyzer().analyze({})

    def test_to_dict_roundtrip_shape(self):
        results = {"eye_contact": _result("eye_contact", 40)}
        assessment = Analyzer().analyze(results, frames_analyzed=10, frames_with_speaker=9)
        d = assessment.to_dict()
        assert d["grade"] == assessment.grade
        assert "eye_contact" in d["metrics"]
        assert isinstance(d["recommendations"], list)
        assert d["recommendations"][0]["metric"] == "eye_contact"
