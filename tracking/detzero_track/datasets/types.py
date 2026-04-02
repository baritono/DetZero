"""
Typed schema definitions for the tracking data pipeline.

These TypedDict classes document the per-frame detection and tracking
structures that flow through the tracking module.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# DetFrameInfo
# ---------------------------------------------------------------------------

class DetFrameInfo(TypedDict, total=False):
    """
    Per-frame detection information for one frame within a driving sequence.

    Loaded from the serialised detection results (pickle file) and processed
    by :class:`detzero_track.datasets.data_processor.DataProcessor`.

    Fields
    ------
    sequence_name : str
        Identifier of the driving sequence this frame belongs to.
    frame_id : int
        Zero-based frame index within the sequence.
    timestamp : int
        Microsecond timestamp of the frame.
    pose : np.ndarray, shape (4, 4), dtype float64
        LIDAR-to-world rigid-body transformation matrix.
    boxes_lidar : np.ndarray, shape (N, 7), dtype float32
        Detected bounding boxes in the LIDAR frame.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
        ``N`` = number of detections in this frame (0 when the frame is
        empty).
    score : np.ndarray, shape (N,), dtype float32
        Detection confidence score for each box.
    name : np.ndarray, shape (N,), dtype object (str)
        Class name for each box: ``'Vehicle'``, ``'Pedestrian'``,
        or ``'Cyclist'``.
    num_points : np.ndarray, shape (N,), dtype int32
        Number of LIDAR points inside each detection box.
        Populated by the ``points_in_box`` data-processor step.
    """

    sequence_name: str
    frame_id: int
    timestamp: int
    pose: np.ndarray        # (4, 4)  float64
    boxes_lidar: np.ndarray  # (N, 7) float32
    score: np.ndarray       # (N,)    float32
    name: np.ndarray        # (N,)    str
    num_points: np.ndarray  # (N,)    int32


# ---------------------------------------------------------------------------
# GtFrameInfo
# ---------------------------------------------------------------------------

class GtFrameInfo(TypedDict, total=False):
    """
    Per-frame ground-truth annotation for evaluation / track assignment.

    Loaded from ``waymo_infos_<split>.pkl``.

    Fields
    ------
    sequence_name : str
        Driving sequence identifier.
    frame_id : int
        Frame index within the sequence.
    timestamp : int
        Microsecond timestamp.
    pose : np.ndarray, shape (4, 4), dtype float64
        LIDAR-to-world transformation.
    gt_boxes_lidar : np.ndarray, shape (M, 7), dtype float32
        Ground-truth bounding boxes in the LIDAR frame.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
    gt_names : np.ndarray, shape (M,), dtype object (str)
        Class name for each GT box.
    obj_ids : np.ndarray, shape (M,), dtype object (str)
        Unique object identifier (track ID) for each GT box.
    """

    sequence_name: str
    frame_id: int
    timestamp: int
    pose: np.ndarray            # (4, 4)  float64
    gt_boxes_lidar: np.ndarray  # (M, 7)  float32
    gt_names: np.ndarray        # (M,)    str
    obj_ids: np.ndarray         # (M,)    str


# ---------------------------------------------------------------------------
# TrackingBatchDict
# ---------------------------------------------------------------------------

class TrackingBatchDict(TypedDict, total=False):
    """
    Batch dictionary returned by :meth:`WaymoTrackDataset.collate_batch`.

    Because the tracking dataset processes one *sequence* at a time, the
    "batch" here is effectively a list of sequences rather than a list of
    individual frames.

    Fields
    ------
    detection : List[Dict[int, DetFrameInfo]]
        One entry per sequence.  Each entry maps frame index to the
        processed detection info for that frame.
    det_drop : List[Dict[int, DetFrameInfo]]
        Low-confidence detections filtered out during data processing;
        same structure as ``detection``.
    gt : List[Dict[int, GtFrameInfo]]
        Ground-truth annotations per sequence (only present in evaluation
        / training mode where ``assign_mode`` is ``True``).
    """

    detection: List[Dict[int, DetFrameInfo]]
    det_drop: List[Dict[int, DetFrameInfo]]
    gt: List[Dict[int, GtFrameInfo]]


# ---------------------------------------------------------------------------
# TrackResultEntry
# ---------------------------------------------------------------------------

class TrackResultEntry(TypedDict):
    """
    Single tracked-object entry written to the output pickle by the tracker.

    Fields
    ------
    sequence_name : str
        Sequence identifier.
    obj_id : str
        Unique track identifier assigned by the tracker.
    name : str
        Object class: ``'Vehicle'``, ``'Pedestrian'``, or ``'Cyclist'``.
    frame_id : List[int]
        Ordered list of frame indices where this track was active.
    boxes_lidar : List[np.ndarray]
        Per-frame bounding box in the LIDAR frame, each shape ``(7,)``:
        ``[x, y, z, dx, dy, dz, yaw]``.
    score : List[float]
        Per-frame detection / track confidence score.
    pose : List[np.ndarray]
        Per-frame LIDAR-to-world transform, each shape ``(4, 4)``.
    """

    sequence_name: str
    obj_id: str
    name: str
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]   # each (7,) float32
    score: List[float]
    pose: List[np.ndarray]          # each (4, 4) float64
