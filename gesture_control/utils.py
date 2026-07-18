"""Shared utility functions for the gesture control pipeline."""

from typing import List

# Number of hand landmarks MediaPipe produces
NUM_LANDMARKS = 21
# Feature vector length: x, y, z for each landmark
NUM_FEATURES = NUM_LANDMARKS * 3  # 63


def flatten_landmarks(
    landmarks: List[object],
) -> List[float]:
    """Flatten a list of MediaPipe landmark objects into a 63-D feature vector.

    Each landmark contributes (x, y, z) — normalised coordinates in [0, 1].
    """
    features: list[float] = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])
    return features


def label_from_action(action_name: str) -> str:
    """Normalise a gesture label for use as a filename-friendly token."""
    return action_name.strip().upper().replace(" ", "_")
