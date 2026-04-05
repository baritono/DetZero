"""
Typed data-structure definitions for the DetZero tracking pipeline.

Every dictionary that flows between tracking pipeline components is defined
here as a :class:`typing.TypedDict` so that:

* the field names are explicit and discoverable,
* the tensor/array shapes and dtypes are documented in the docstring of each
  field,
* static type-checkers (mypy, pyright) can verify correctness, and
* IDEs can provide auto-complete for field names.

Naming conventions
------------------
- Fields that exist in more than one TypedDict and carry the same semantic
  meaning use **exactly the same field name** (e.g. ``boxes_global``,
  ``boxes_lidar``, ``name``, ``score``, ``pose``, ``sequence_name``,
  ``frame_id``, ``sample_idx``, ``obj_ids``).
- Shape annotations use abbreviated dimension names:
    ``N`` – number of detected objects in a single frame,
    ``T`` – number of time-steps (frames) in a tracklet.
- :class:`AnnotationDict` is intentionally imported from the shared
  :mod:`detzero_utils.structures` module so that the detection→tracking
  interface type is defined in exactly one place.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
from typing_extensions import TypedDict

# The detection→tracking interface type lives in the shared utilities package.
from detzero_utils.structures import AnnotationDict  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Per-frame detection data flowing into the tracking pipeline
# ---------------------------------------------------------------------------

class FrameDetectionData(AnnotationDict, total=False):
    """Per-frame detection results as consumed by the tracking pipeline.

    Extends :class:`detzero_utils.structures.AnnotationDict` with
    tracking-specific fields that are added during pre-processing.

    Base fields (inherited from :class:`AnnotationDict`)
    -----------------------------------------------------
    name : np.ndarray, shape (N,), dtype str
        Class name for each detected object.
    score : np.ndarray, shape (N,), dtype float32
        Confidence score in ``[0, 1]`` for each detection.
    boxes_lidar : np.ndarray, shape (N, 7) or (N, 9), dtype float32
        Bounding boxes in the ego-vehicle (LiDAR) frame:
        ``[x, y, z, dx, dy, dz, heading]``, optionally with ``[vx, vy]``.
    sequence_name : str
        Identifier of the driving sequence.
    frame_id : int or str
        Frame identifier within the sequence.
    pose : np.ndarray, shape (4, 4), dtype float64
        Ego-to-world SE(3) transform matrix.

    Additional tracking fields (added by :class:`DataProcessor`)
    -------------------------------------------------------------
    boxes_global : np.ndarray, shape (N, 7) or (N, 9), dtype float32
        Bounding boxes transformed into the world (global) coordinate frame
        using ``pose``.  Added by the ``transform_to_global`` processor step.
    num_points : np.ndarray, shape (N,), dtype int32
        Number of LiDAR points inside each detection box.  Added by the
        ``points_in_box`` processor step.
    sample_idx : int or str
        Alias for ``frame_id`` used by the tracking codebase.  Some detection
        result files use ``sample_idx`` as the key; both refer to the same
        frame identifier within the sequence.
    timestamp : float
        Optional UNIX timestamp (seconds) for the frame.
    """

    boxes_global: np.ndarray
    num_points: np.ndarray
    sample_idx: Union[str, int]
    timestamp: float


# ---------------------------------------------------------------------------
# Per-track state at a single time-step
# ---------------------------------------------------------------------------

class TrackFrameState(TypedDict):
    """State of a single track at a single frame.

    Produced by :meth:`BaseKalmanFilter.info` and accumulated in
    :meth:`TrackManager.online_track_module` to build the full
    :class:`TrackletData` history.

    Fields
    ------
    boxes_global : np.ndarray, shape (9,), dtype float32
        Track state as a bounding box in the global frame:
        ``[x, y, z, dx, dy, dz, heading, vx, vy]``.  The last two entries
        are the velocity estimated by the Kalman filter.
    name : str
        Class name of the tracked object (e.g. ``'Vehicle'``).
    score : float
        Latest detection confidence score associated with this track.
    sample_idx : int or str
        Frame identifier at which this state was recorded.
    hit : int
        Hit status for this time-step:
        ``0`` – the track was not matched (predicted/coasted frame),
        ``1`` – matched by first-stage association,
        ``2`` – matched by second-stage (low-confidence) association.
    num_points : int or float
        Number of LiDAR points inside the matched detection box at this
        time-step.  ``0`` when the track was not matched.
    obj_ids : int
        Unique integer track identifier assigned at track birth.
    """

    boxes_global: np.ndarray
    name: str
    score: float
    sample_idx: Union[str, int]
    hit: int
    num_points: Union[int, float]
    obj_ids: int


# ---------------------------------------------------------------------------
# Full tracklet history (output of the tracker)
# ---------------------------------------------------------------------------

class TrackletDataBase(TypedDict):
    """Required fields of a tracklet's accumulated history.

    All arrays have the same leading dimension ``T`` (the number of frames
    in the tracklet's lifespan).
    """

    boxes_global: np.ndarray
    """Bounding-box states over time, shape ``(T, 9)``, dtype float32.

    Each row is ``[x, y, z, dx, dy, dz, heading, vx, vy]`` in the global
    coordinate frame.
    """

    name: np.ndarray
    """Class name at each time-step, shape ``(T,)``, dtype str (object).

    Typically constant across the track's lifetime, but may change if the
    detector's class prediction flips.
    """

    score: np.ndarray
    """Detection confidence at each time-step, shape ``(T,)``, dtype float32.

    For coasted frames (``hit == 0``) this is the score from the most recent
    matched detection.
    """

    sample_idx: np.ndarray
    """Frame identifiers for each time-step, shape ``(T,)``, dtype int-like.

    Stores the ``frame_id`` / ``sample_idx`` value from
    :class:`FrameDetectionData` for each frame the track was active.
    """

    hit: np.ndarray
    """Hit flags per time-step, shape ``(T,)``, dtype int.

    ``0`` = coasted (no matched detection),
    ``1`` = matched (first-stage),
    ``2`` = matched (second-stage / low-confidence).
    """

    num_points: np.ndarray
    """LiDAR point counts per time-step, shape ``(T,)``, dtype int.

    Number of points inside the matched detection box; ``0`` for coasted
    frames.
    """

    obj_ids: np.ndarray
    """Track identifiers per time-step, shape ``(T,)``, dtype int.

    All elements are equal to the track's birth-time ``track_id``.
    Stored as an array for convenience when converting between per-object
    and per-frame representations.
    """

    pose: np.ndarray
    """Ego-to-world SE(3) transforms per time-step, shape ``(T, 4, 4)``,
    dtype float64.

    Copied from :attr:`FrameDetectionData.pose` for the corresponding frame.
    """


class TrackletData(TrackletDataBase, total=False):
    """Full tracklet history produced by the tracker.

    Extends :class:`TrackletDataBase` with optional fields that are added
    during post-processing or target assignment.

    Optional fields
    ---------------
    state : str
        Motion state classification, either ``'static'`` or ``'dynamic'``.
        Set by :meth:`PostProcessor.motion_classify`.
    iou : np.ndarray, shape (T,), dtype float32
        Per-frame IoU between this track and its assigned ground-truth
        object.  Set by :func:`assign_track_target`.
    iou_idx : list of int
        Temporary list of row/column indices into the per-frame IoU matrix.
        Used internally by :func:`assign_track_target` and
        :meth:`TrackRecall.eval_single_seq`; removed before the tracklet
        dict is returned.
    start : np.ndarray, shape (T,), dtype int
        Binary flag; ``1`` on the first frame of the track's life, ``0``
        otherwise.  Used during the reverse-tracking pass in
        :meth:`TrackManager.reverse_tracking_module`.
    """

    state: str
    iou: np.ndarray
    iou_idx: List[int]
    start: np.ndarray


# ---------------------------------------------------------------------------
# Ground-truth annotation dicts (Waymo evaluation / target assignment)
# ---------------------------------------------------------------------------

class GroundTruthAnnotations(TypedDict, total=False):
    """Nested annotations sub-dict inside each ground-truth frame record.

    Stored under the ``'annos'`` key of :class:`GroundTruthFrameData`.
    Populated from the Waymo ``waymo_infos_*.pkl`` files.

    Fields
    ------
    name : np.ndarray, shape (N,), dtype str
        Class name for each annotated object (e.g. ``'Vehicle'``).
    obj_ids : np.ndarray, shape (N,), dtype int-like
        Unique per-sequence object identifiers assigned by the dataset.
    gt_boxes_lidar : np.ndarray, shape (N, 7+), dtype float32
        Ground-truth bounding boxes in the LiDAR frame:
        ``[x, y, z, dx, dy, dz, heading, ...]``.
    gt_boxes_global : np.ndarray, shape (N, 7+), dtype float32
        Ground-truth bounding boxes transformed into the world frame using
        the frame's ``pose``.
    difficulty : np.ndarray, shape (N,), dtype int32
        Per-object difficulty label (``0`` = unknown, ``1`` = L1, ``2`` = L2).
    num_points_in_gt : np.ndarray, shape (N,), dtype int32
        Number of LiDAR points inside each ground-truth box.
    """

    name: np.ndarray
    obj_ids: np.ndarray
    gt_boxes_lidar: np.ndarray
    gt_boxes_global: np.ndarray
    difficulty: np.ndarray
    num_points_in_gt: np.ndarray


class GroundTruthFrameData(TypedDict, total=False):
    """Per-frame ground-truth record loaded from ``waymo_infos_*.pkl``.

    Used by :func:`assign_track_target`, :class:`TrackRecall`, and
    :func:`get_iou_mat_dict` to evaluate tracking quality and assign GT
    labels to predicted tracklets.

    Fields
    ------
    annos : GroundTruthAnnotations
        Nested annotations sub-dict; see :class:`GroundTruthAnnotations`.
    sequence_name : str
        Identifier of the driving sequence.
    frame_id : int or str
        Frame identifier within the sequence.
    sample_idx : int or str
        Alias for ``frame_id`` (some annotation files use this key).
    pose : np.ndarray, shape (4, 4), dtype float64
        Ego-to-world SE(3) transform for this frame.
    """

    annos: GroundTruthAnnotations
    sequence_name: str
    frame_id: Union[str, int]
    sample_idx: Union[str, int]
    pose: np.ndarray


# ---------------------------------------------------------------------------
# Public API of this module
# ---------------------------------------------------------------------------

__all__ = [
    # Re-exported from detzero_utils (detection→tracking interface)
    "AnnotationDict",
    # Tracking-specific types
    "FrameDetectionData",
    "TrackFrameState",
    "TrackletDataBase",
    "TrackletData",
    "GroundTruthAnnotations",
    "GroundTruthFrameData",
]
