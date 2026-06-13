"""OpenCV-backed video reading, writing, and annotation (I/O edge)."""

from .annotator import FrameAnnotator
from .reader import OpenCVVideoReader, VideoReadError
from .writer import OpenCVVideoWriter

__all__ = [
    "FrameAnnotator",
    "OpenCVVideoReader",
    "VideoReadError",
    "OpenCVVideoWriter",
]
