"""Semantic names for the MediaPipe landmark indices we rely on.

Centralizing these keeps the metric code readable and makes the MediaPipe
dependency explicit in exactly one place. References:

* Face Mesh canonical 468-point topology (+ iris points 468-477).
* Pose 33-point "BlazePose" topology.
"""

from __future__ import annotations


class FaceMesh:
    """Key indices into the 468/478-point Face Mesh."""

    NOSE_TIP = 1
    CHIN = 152
    FOREHEAD_TOP = 10

    # Outer face edges (used for head-width / yaw estimation).
    LEFT_CHEEK = 234   # subject's right side as seen, image-left
    RIGHT_CHEEK = 454  # subject's left side, image-right

    # Eye corners (horizontal extent) and lids (vertical extent) for EAR.
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Iris centers (only present when refine_landmarks/irises enabled).
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Eyebrow centers (vertical brow-raise expressivity).
    LEFT_BROW = 105
    RIGHT_BROW = 334

    # Mouth corners and lips (vertical opening / smile width).
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    UPPER_LIP = 13
    LOWER_LIP = 14


class Pose:
    """The 33 BlazePose landmark indices, by name."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    NUM_LANDMARKS = 33


class Hands:
    """The 21 MediaPipe Hands landmark indices, by name (one set per hand)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    NUM_LANDMARKS = 21

    #: Fingertip indices, handy for openness / pointing analysis.
    FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

    #: Bone connections for drawing a hand skeleton.
    CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (5, 9), (9, 10), (10, 11), (11, 12),     # middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 17),                                  # palm base
    )

