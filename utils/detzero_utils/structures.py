"""
Typed data-structure definitions shared between the DetZero detection and
tracking pipelines.

Types defined here form the *interface* between the two modules: the detection
module produces :class:`AnnotationDict` records and the tracking module consumes
them as its primary per-frame input.

Naming conventions
------------------
- Fields that exist in more than one TypedDict and carry the same semantic
  meaning use **exactly the same field name** (e.g. ``name``, ``score``,
  ``boxes_lidar``, ``sequence_name``, ``frame_id``, ``pose``).
- Shape annotations use abbreviated dimension names:
    ``N`` – number of detected objects in the frame,
    ``T`` – number of time-steps in a track.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from typing_extensions import TypedDict


class AnnotationDict(TypedDict, total=False):
    """Per-frame detection output, shared between the detection and tracking modules.

    Produced by :meth:`WaymoDataset.generate_prediction_dicts` (detection
    side) and consumed by :class:`WaymoTrackDataset` (tracking side) as the
    primary per-frame input to the tracking pipeline.  Also used by the Waymo
    evaluation toolkit.

    All fields are optional at the TypedDict level because the dict is built
    incrementally; callers should ensure the fields they need are present.

    Fields
    ------
    name : np.ndarray, shape (N,), dtype str
        Predicted class name for each detection (e.g. ``'Vehicle'``,
        ``'Pedestrian'``, ``'Cyclist'``).
    score : np.ndarray, shape (N,), dtype float32
        Detection confidence score in ``[0, 1]`` for each object.
    boxes_lidar : np.ndarray, shape (N, 7) or (N, 9), dtype float32
        Predicted 3-D bounding boxes in the ego-vehicle (LiDAR) coordinate
        frame: ``[x, y, z, dx, dy, dz, heading]`` (7 columns), optionally
        extended with velocity ``[vx, vy]`` (9 columns).
    sequence_name : str
        Identifier of the driving sequence this frame belongs to
        (e.g. the Waymo segment name).
    frame_id : int or str
        Unique identifier for the LiDAR frame within its sequence.
    pose : np.ndarray, shape (4, 4), dtype float64
        Ego-to-world SE(3) transform matrix for this frame.
    """

    name: np.ndarray
    score: np.ndarray
    boxes_lidar: np.ndarray
    sequence_name: str
    frame_id: Union[str, int]
    pose: np.ndarray


__all__ = [
    "AnnotationDict",
]
