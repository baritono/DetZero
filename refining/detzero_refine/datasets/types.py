"""
Typed schema definitions for the refinement data pipeline.

These TypedDict classes document the structures that flow through the
geometry, position, and confidence refinement modules.
"""

from __future__ import annotations

from typing import List, Optional
from typing import TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# RawTrackInfo
# ---------------------------------------------------------------------------

class RawTrackInfo(TypedDict, total=False):
    """
    Raw per-tracklet information loaded from a sequence pickle file by
    :meth:`DatasetTemplate.load_track_infos`.

    One entry represents a complete object track across the frames of a
    sequence in which it was detected.

    Fields
    ------
    name : str
        Class name of the tracked object: ``'Vehicle'``, ``'Pedestrian'``,
        or ``'Cyclist'``.
    boxes_global : np.ndarray, shape (T, 7), dtype float32
        Detection boxes in the *global* (world) coordinate frame for each of
        the ``T`` frames where the object was detected.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
    gt_boxes_global : np.ndarray, shape (T, 7), dtype float32
        Matched ground-truth boxes in the global frame.
        Present only during training (when a GT match exists).
    score : np.ndarray, shape (T,), dtype float32
        Detection confidence score for each frame.
    sample_idx : np.ndarray, shape (T,), dtype int32
        Frame index within the sequence for each detection.
    pose : np.ndarray, shape (T, 4, 4), dtype float64
        LIDAR-to-world transformation matrix for each frame.
    pts : List[np.ndarray]
        List of ``T`` point-cloud arrays.  The ``i``-th element has shape
        ``(N_i, C)`` where ``N_i`` varies per frame and ``C ≥ 4``
        (``[x, y, z, intensity, ...]``).
    matched : np.ndarray, shape (T,), dtype bool
        ``True`` for frames where the detection was matched to a GT box.
    matched_tracklet : bool
        ``True`` when the entire tracklet has a corresponding GT track.
    state : np.ndarray, shape (T,), dtype int32
        Tracking state code for each frame (detector-specific).
    """

    name: str
    boxes_global: np.ndarray    # (T, 7)        float32
    gt_boxes_global: np.ndarray  # (T, 7)       float32  — training only
    score: np.ndarray           # (T,)           float32
    sample_idx: np.ndarray      # (T,)           int32
    pose: np.ndarray            # (T, 4, 4)      float64
    pts: List[np.ndarray]       # T × (N_i, C)
    matched: np.ndarray         # (T,)           bool
    matched_tracklet: bool
    state: np.ndarray           # (T,)           int32


# ---------------------------------------------------------------------------
# RefineDataDict
# ---------------------------------------------------------------------------

class RefineDataDict(TypedDict, total=False):
    """
    Processed tracklet feature dict fed into the refinement models.

    Produced by :meth:`WaymoGeometryDataset.extract_track_feature`,
    :meth:`WaymoPositionDataset.extract_track_feature`, or
    :meth:`WaymoConfidenceDataset.extract_track_feature` and then passed
    through :meth:`DatasetTemplate.prepare_data`.

    Common fields (all three refinement tasks)
    ------------------------------------------
    obj_cls : int
        1-based class index (1 = Vehicle, 2 = Pedestrian, 3 = Cyclist).
    traj : np.ndarray, shape (T_sub, 7), dtype float32
        Subsampled + transformed detection trajectory.
        In geometry / confidence refinement the coordinate origin is the
        first sampled box; in position refinement it is a randomly chosen
        box.  Heading and centre are zeroed out after transformation.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
    traj_gt : np.ndarray, shape (T_sub, 7), dtype float32
        Corresponding GT trajectory in the same coordinate system.
        Present only during training.
    score : np.ndarray, shape (T_sub,), dtype float32
        Detection scores for the subsampled frames.
    pose : np.ndarray, shape (T_sub, 4, 4), dtype float64
        Per-frame LIDAR-to-world transforms for the subsampled frames.
    frm_id : np.ndarray, shape (T_sub,), dtype int32
        Frame indices within the sequence for the subsampled frames.
    pts : np.ndarray, shape (N_total, C_feat), dtype float32
        Encoded point features concatenated across all subsampled frames.
        ``C_feat`` depends on the ``ENCODING`` config (xyz, intensity,
        point-to-surface distances, score, etc.).

    Geometry-refinement-specific fields
    ------------------------------------
    query_pts : List[np.ndarray]
        Point clouds for ``query_num`` selected query proposals,
        each shape ``(N_q, C_feat)``.
    query_box : np.ndarray, shape (query_num, 7), dtype float32
        Boxes for the query proposals (centre / heading zeroed).
    gt_box : np.ndarray, shape (query_num, 7), dtype float32
        GT boxes corresponding to the query proposals.

    Position-refinement-specific fields
    ------------------------------------
    init_box : np.ndarray, shape (7,), dtype float32
        The initial coordinate origin box (centre / heading zeroed).
    query_pts : List[np.ndarray]
        Point clouds for the selected query proposals.
    query_traj : np.ndarray, shape (query_num, 7), dtype float32
        Trajectory boxes for the query proposals.

    Confidence-refinement-specific fields
    ---------------------------------------
    matched_tracklet : bool
        Whether the tracklet is a positive (matched) or negative sample.
    """

    obj_cls: int
    traj: np.ndarray            # (T_sub, 7)            float32
    traj_gt: np.ndarray         # (T_sub, 7)            float32 — training only
    score: np.ndarray           # (T_sub,)              float32
    pose: np.ndarray            # (T_sub, 4, 4)         float64
    frm_id: np.ndarray          # (T_sub,)              int32
    pts: np.ndarray             # (N_total, C_feat)     float32
    query_pts: List[np.ndarray]  # query_num × (N_q, C_feat)
    query_box: np.ndarray       # (query_num, 7)        float32
    gt_box: np.ndarray          # (query_num, 7)        float32
    init_box: np.ndarray        # (7,)                  float32
    query_traj: np.ndarray      # (query_num, 7)        float32
    matched_tracklet: bool


# ---------------------------------------------------------------------------
# RefineAnnoDictEntry
# ---------------------------------------------------------------------------

class RefineAnnoDictEntry(TypedDict):
    """
    Per-object refinement result entry written into the output prediction
    dictionary by :meth:`WaymoGeometryDataset.generate_prediction_dicts`.

    Fields
    ------
    sequence_name : str
        Sequence identifier.
    frame_id : List[int]
        Frame indices for each refined box.
    boxes_lidar : List[np.ndarray]
        Refined bounding boxes in the LIDAR frame, one ``(7,)`` array per
        frame.  Each: ``[x, y, z, dx, dy, dz, yaw]``.
    score : List[float]
        Detection confidence score per frame.
    name : List[str]
        Class name per frame (e.g. ``'Vehicle'``).
    pose : List[np.ndarray]
        LIDAR-to-world transform per frame, each shape ``(4, 4)``.
    """

    sequence_name: str
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]  # each (7,) float32
    score: List[float]
    name: List[str]
    pose: List[np.ndarray]         # each (4, 4) float64
