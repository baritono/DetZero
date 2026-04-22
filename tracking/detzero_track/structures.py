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

from typing import Any, Callable, Dict, List, Union

import numpy as np
from typing_extensions import TypedDict

# The detection→tracking interface type lives in the shared utilities package.
from detzero_utils.structures import AnnotationDict, FrameId  # noqa: F401  (re-exported)


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

    boxes_global: np.ndarray  # shape: (N, 7) or (N, 9) – bounding boxes in world/global frame; columns: [x, y, z, dx, dy, dz, heading] (meters for x/y/z/dx/dy/dz, radians for heading, in (-π, π]); optional last 2 cols: [vx, vy] in m/s
    num_points: np.ndarray  # shape: (N,) – number of LiDAR points inside each detection box; dtype int32; 0 for unmatched tracks
    sample_idx: FrameId
    timestamp: float
    obj_ids: np.ndarray  # shape: (N,) – sorted track/object IDs for objects active in this frame; dtype int-like
    """Per-frame object/track IDs, shape ``(N,)``, dtype int-like.

    Present after the frame dict is built by :func:`tracklets_to_frames`,
    where it stores the sorted track IDs of all objects active in the frame.
    Also accessed in :func:`assign_track_target` and
    :meth:`TrackRecall.eval_single_seq` to match per-frame objects to
    their corresponding tracklets.
    """


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

    boxes_global: np.ndarray  # shape: (9,) – single-frame track state in world/global frame: [x, y, z, dx, dy, dz, heading, vx, vy]; x/y/z/dx/dy/dz in meters, heading in radians in (-π, π], vx/vy in m/s (Kalman-filter velocity estimate)
    name: str
    score: float
    sample_idx: FrameId
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

    boxes_global: np.ndarray  # shape: (T, 9) – bounding-box history in world/global frame; T = number of frames in tracklet lifespan; columns: [x, y, z, dx, dy, dz, heading, vx, vy]; x/y/z/dx/dy/dz in meters, heading in radians in (-π, π], vx/vy in m/s
    """Bounding-box states over time, shape ``(T, 9)``, dtype float32.

    Each row is ``[x, y, z, dx, dy, dz, heading, vx, vy]`` in the global
    coordinate frame.
    """

    name: np.ndarray  # shape: (T,) – class name at each time-step (e.g. 'Vehicle', 'Pedestrian', 'Cyclist'); dtype object (str)
    """Class name at each time-step, shape ``(T,)``, dtype str (object).

    Typically constant across the track's lifetime, but may change if the
    detector's class prediction flips.
    """

    score: np.ndarray  # shape: (T,) – detection confidence score in [0, 1] at each time-step; dtype float32; coasted frames repeat the most recent matched score
    """Detection confidence at each time-step, shape ``(T,)``, dtype float32.

    For coasted frames (``hit == 0``) this is the score from the most recent
    matched detection.
    """

    sample_idx: np.ndarray  # shape: (T,) – frame identifier for each time-step (matches FrameDetectionData.frame_id); dtype int-like
    """Frame identifiers for each time-step, shape ``(T,)``, dtype int-like.

    Stores the ``frame_id`` / ``sample_idx`` value from
    :class:`FrameDetectionData` for each frame the track was active.
    """

    hit: np.ndarray  # shape: (T,) – association status at each time-step; dtype int; 0 = coasted, 1 = first-stage match, 2 = second-stage (low-confidence) match
    """Hit flags per time-step, shape ``(T,)``, dtype int.

    ``0`` = coasted (no matched detection),
    ``1`` = matched (first-stage),
    ``2`` = matched (second-stage / low-confidence).
    """

    num_points: np.ndarray  # shape: (T,) – LiDAR point count inside matched detection box at each time-step; dtype int; 0 for coasted frames
    """LiDAR point counts per time-step, shape ``(T,)``, dtype int.

    Number of points inside the matched detection box; ``0`` for coasted
    frames.
    """

    obj_ids: np.ndarray  # shape: (T,) – unique integer track ID repeated for each time-step; dtype int; all elements equal to the track's birth-time track_id
    """Track identifiers per time-step, shape ``(T,)``, dtype int.

    All elements are equal to the track's birth-time ``track_id``.
    Stored as an array for convenience when converting between per-object
    and per-frame representations.
    """

    pose: np.ndarray  # shape: (T, 4, 4) – ego-to-world SE(3) homogeneous transform at each time-step; dtype float64; copied from FrameDetectionData.pose for each active frame
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
    iou: np.ndarray  # shape: (T,) – per-frame IoU between this track and its assigned GT object; dtype float32; set by assign_track_target
    iou_idx: List[int]
    start: np.ndarray  # shape: (T,) – binary flag; 1 on the first frame of the track's life, 0 otherwise; dtype int; used during reverse-tracking pass
    boxes_lidar: np.ndarray  # shape: (T, 7) – bounding-box history in ego-vehicle/LiDAR frame; columns: [x, y, z, dx, dy, dz, heading]; x/y/z/dx/dy/dz in meters, heading in radians in (-π, π]; optional field
    """Per-time-step bounding boxes in the LiDAR (ego-vehicle) frame,
    shape ``(T, 7)``, dtype float32: ``[x, y, z, dx, dy, dz, heading]``.

    Optional: only present when the tracklet was created from a source that
    stores lidar boxes (e.g. after :func:`tracklets_to_frames` back-converts
    via :func:`transform_boxes3d`).  Checked with ``'boxes_lidar' in obj_data``
    before access in :func:`tracklets_to_frames`.
    """


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

    name: np.ndarray  # shape: (N,) – class name for each annotated object (e.g. 'Vehicle', 'Pedestrian', 'Cyclist'); dtype object (str); N = number of GT objects in this frame
    obj_ids: np.ndarray  # shape: (N,) – unique per-sequence object identifiers assigned by the Waymo dataset; dtype int-like
    gt_boxes_lidar: np.ndarray  # shape: (N, 7+) – GT bounding boxes in ego-vehicle/LiDAR frame; columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians in (-π, π]
    gt_boxes_global: np.ndarray  # shape: (N, 7+) – GT bounding boxes in world/global frame (transformed via frame pose); columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians in (-π, π]
    difficulty: np.ndarray  # shape: (N,) – per-object difficulty label; dtype int32; 0 = unknown, 1 = L1 (>5 LiDAR points), 2 = L2 (≤5 LiDAR points)
    num_points_in_gt: np.ndarray  # shape: (N,) – number of LiDAR points inside each GT box; dtype int32


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
    frame_id: FrameId
    sample_idx: FrameId
    pose: np.ndarray  # shape: (4, 4) – ego-to-world SE(3) homogeneous transform for this frame; dtype float64


# ---------------------------------------------------------------------------
# Target-assignment output dicts
# ---------------------------------------------------------------------------

class LabeledTrackEntry(TypedDict, total=False):
    """One entry in the labeled portion of the target-assignment result.

    Produced by :func:`assign_track_target` for each track that was
    successfully matched to a ground-truth object.

    Fields
    ------
    track : TrackletData
        The predicted tracklet (with ``iou`` scores added, ``iou_idx``
        removed).
    gt : GroundTruthTrackletData
        The matched ground-truth object's history, with fields from
        ``gt_keys`` (e.g. ``gt_boxes_global``, ``gt_boxes_lidar``,
        ``name``, ``obj_ids``).
    """

    track: TrackletData
    gt: GroundTruthTrackletData


class UnlabeledTrackEntry(TypedDict, total=False):
    """One entry in the unlabeled portion of the target-assignment result.

    Produced by :func:`assign_track_target` for each track that could not
    be matched to any ground-truth object.

    Fields
    ------
    track : TrackletData
        The predicted tracklet (``iou_idx`` removed).
    """

    track: TrackletData


class AssignTargetResult(TypedDict):
    """Return value of :func:`assign_track_target`.

    Fields
    ------
    label : dict mapping track_id (int) to LabeledTrackEntry
        Tracks that were matched to a ground-truth object.
    unlabel : dict mapping track_id (int) to UnlabeledTrackEntry
        Tracks that could not be matched to any ground-truth object.
    """

    label: Dict[int, LabeledTrackEntry]
    unlabel: Dict[int, UnlabeledTrackEntry]


# ---------------------------------------------------------------------------
# TrackRecall evaluation input dict
# ---------------------------------------------------------------------------

class EvalSequenceInput(TypedDict):
    """Input dict for :meth:`TrackRecall.eval_single_seq`.

    Fields
    ------
    gt : dict mapping frame_id (str) to GroundTruthFrameData
        Ground-truth frame records for the sequence.
    pred : dict mapping track_id (int) to TrackletData
        Predicted tracklet data for the sequence.
    """

    gt: Dict[str, GroundTruthFrameData]
    pred: Dict[int, TrackletData]


# ---------------------------------------------------------------------------
# Ground-truth object tracklet (output of get_gt_id_data)
# ---------------------------------------------------------------------------

class GroundTruthTrackletData(TypedDict, total=False):
    """GT object data indexed by GT object ID, produced by :func:`get_gt_id_data`.

    All fields are stored as lists while the data is being accumulated and then
    converted to numpy arrays in-place (``np.array(gt_id_data[gt_id][key])``).
    The type annotations reflect the post-conversion (array) state, which is
    the state seen by all downstream consumers.  All fields are optional
    because the exact set present depends on the ``gt_keys`` argument passed
    to :func:`get_gt_id_data`.

    Fields
    ------
    sample_idx : np.ndarray, shape (T,), dtype str
        Frame IDs of every appearance of this GT object.
    iou_idx : np.ndarray, shape (T,), dtype int
        Row indices into the per-frame IoU matrix built by
        :func:`get_iou_mat_dict`.  Temporary field; popped after use.
    name : np.ndarray, shape (T,), dtype str
        Class name for each appearance (e.g. ``'Vehicle'``).
    obj_ids : np.ndarray, shape (T,)
        Per-appearance GT object identifiers (all equal for one GT track).
    gt_boxes_global : np.ndarray, shape (T, 7+), dtype float32
        GT bounding boxes in the world (global) coordinate frame.
    gt_boxes_lidar : np.ndarray, shape (T, 7+), dtype float32
        GT bounding boxes in the LiDAR (ego-vehicle) coordinate frame.
    difficulty : np.ndarray, shape (T,), dtype int32
        Per-appearance difficulty label; ``0`` = unknown, ``1`` = L1,
        ``2`` = L2.
    num_points_in_gt : np.ndarray, shape (T,), dtype int32
        Number of LiDAR points inside the GT box at each appearance.
    """

    sample_idx: np.ndarray
    iou_idx: np.ndarray
    name: np.ndarray
    obj_ids: np.ndarray
    gt_boxes_global: np.ndarray
    gt_boxes_lidar: np.ndarray
    difficulty: np.ndarray
    num_points_in_gt: np.ndarray


# ---------------------------------------------------------------------------
# TrackManager internal modules dict
# ---------------------------------------------------------------------------

class TrackManagerModules(TypedDict, total=False):
    """Internal state dict of :class:`TrackManager`, keyed by module name.

    Built incrementally by the ``build_*`` methods and stored as
    ``self.modules_dicts``.  Each field holds either a constructed module
    object, a callable, or an :class:`easydict.EasyDict` config snapshot.

    Fields
    ------
    filter_module : callable
        Partially-applied Kalman-filter constructor; called with ``bbox``,
        ``name``, ``score``, ``frame_id``, ``track_id``, ``num_points`` to
        create a new track.
    filter_config : Any (EasyDict)
        Lower-cased configuration snapshot for the filter
        (e.g. ``name``, ``x_dim``, ``z_dim``, ``delta_t`` …).
    track_age_config : Any (EasyDict)
        Lower-cased configuration for birth/death thresholds
        (``birth_age``, ``death_age``).
    data_association_module : associate_det_to_tracks
        Instantiated data-association object.
    data_association_config : Any (EasyDict)
        Lower-cased configuration snapshot for data association.
    track_merge_config : Any (EasyDict)
        Lower-cased configuration for track merging; includes
        ``enable`` (bool) and ``class_threshold`` (dict mapping class
        name to overlap threshold).
    reverse_tracking_config : Any (EasyDict)
        Lower-cased configuration for the reverse-tracking pass;
        includes ``enable`` (bool).
    """

    filter_module: Callable[..., Any]
    """Partially-applied Kalman-filter constructor (``functools.partial``);
    called with ``bbox``, ``name``, ``score``, ``frame_id``, ``track_id``,
    ``num_points`` to create a new track object."""

    filter_config: Any
    """Lower-cased configuration snapshot for the filter
    (``name``, ``x_dim``, ``z_dim``, ``delta_t`` …).  An
    :class:`easydict.EasyDict` at runtime."""

    track_age_config: Any
    """Lower-cased configuration for birth/death thresholds
    (``birth_age``, ``death_age``).  An :class:`easydict.EasyDict`."""

    data_association_module: Any
    """Instantiated :class:`associate_det_to_tracks` object."""

    data_association_config: Any
    """Lower-cased configuration snapshot for data association.
    An :class:`easydict.EasyDict`."""

    track_merge_config: Any
    """Lower-cased configuration for track merging; includes
    ``enable`` (bool) and ``class_threshold`` (dict mapping class
    name to overlap threshold).  An :class:`easydict.EasyDict`."""

    reverse_tracking_config: Any
    """Lower-cased configuration for the reverse-tracking pass;
    includes ``enable`` (bool).  An :class:`easydict.EasyDict`."""


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
    "GroundTruthTrackletData",
    "LabeledTrackEntry",
    "UnlabeledTrackEntry",
    "AssignTargetResult",
    "EvalSequenceInput",
    "TrackManagerModules",
]
