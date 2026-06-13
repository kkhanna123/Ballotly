"""Report renderer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..analysis.aggregator import OverallAssessment


class ReportRenderer(ABC):
    """Renders an :class:`OverallAssessment` into a single text document."""

    #: File extension (without dot) the rendered content should be saved as.
    extension: str = "txt"

    @abstractmethod
    def render(self, assessment: OverallAssessment) -> str:
        """Return the rendered document as a string."""
