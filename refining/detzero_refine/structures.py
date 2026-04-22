"""Typed data schemas for dictionaries used in the refining pipeline.

This module defines explicit :class:`typing.TypedDict` contracts for the
dataset/model dictionaries passed around under ``refining/`` so that magic
string keys are replaced with declared schemas documented with semantic intent.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch
from typing_extensions import TypedDict

from detzero_utils.structures import FrameId, PoseMatrix, SequenceName

NumpyOrTorchArray = Union[np.ndarray, torch.Tensor]
TrackId = Union[int, str]


class TrackObjectInfo(TypedDict, total=False):
    """Object-track data loaded from refining input pickle files.

    ``T`` denotes the track length (number of frames for this object track).
    """

    sequence_name: SequenceName
    obj_id: TrackId
    name: str
    sample_idx: np.ndarray  # shape: (T,) – frame IDs for each step of this object track; T = track length; dtype int-like
    """Frame IDs for this object track, shape ``(T,)``."""
    boxes_global: np.ndarray  # shape: (T, 7+) – track boxes in world/global frame; columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians in (-pi, pi]; optional extra columns (e.g. vx, vy in m/s)
    """Track boxes in world frame, shape ``(T, 7+)``: ``[x, y, z, dx, dy, dz, heading, ...]``."""
    gt_boxes_global: np.ndarray  # shape: (T, 7+) – matched GT boxes in world/global frame; same column layout as boxes_global; x/y/z/dx/dy/dz in meters, heading in radians
    """Matched GT boxes in world frame, shape ``(T, 7+)``."""
    score: np.ndarray  # shape: (T,) – detection confidence score in [0, 1] for each frame step; dtype float32
    """Detection scores per frame, shape ``(T,)``."""
    pose: np.ndarray  # shape: (T, 4, 4) – ego-to-world SE(3) homogeneous transform for each frame step; dtype float64
    """Ego-to-world transforms, shape ``(T, 4, 4)``."""
    pts: List[np.ndarray]  # list of length T; each element shape (Pi, C) – cropped LiDAR point cloud for step i in the local box-center frame; Pi = variable point count, C = feature channels (x, y, z, intensity, ...)
    """Per-frame cropped point clouds, list length ``T``, each shape ``(Pi, C)``."""
    matched: np.ndarray  # shape: (T,) – per-frame proposal-to-GT match flag; dtype bool; True when the predicted box was matched to a GT box
    """Per-frame proposal-to-GT match flag, shape ``(T,)`` bool."""
    matched_tracklet: bool
    state: str
    """Track motion state (e.g. ``'static'`` or ``'dynamic'``)."""
    refine_iou: np.ndarray  # shape: (T,) – per-frame IoU supervision target used by confidence refining; float32 in [0, 1]; -1 sentinel for missing/unmatched frames
    """Per-frame IoU target used by confidence refining, shape ``(T,)``."""


class GeometrySampleDict(TypedDict, total=False):
    """Per-object sample dictionary for geometry refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray  # shape: (Q,) – frame IDs for the Q selected query proposals; dtype int-like
    obj_id: TrackId
    obj_cls: int
    geo_query_num: int
    geo_query_boxes: np.ndarray  # shape: (Q, 7) – query boxes in local box-center frame with center and heading zeroed out; columns: [0, 0, 0, dx, dy, dz, 0]; dx/dy/dz in meters; Q = number of query proposals (up to QUERY_NUM)
    """Query boxes in local normalized frame, shape ``(Q, 7)``."""
    geo_query_points: Union[np.ndarray, List[np.ndarray]]  # shape: (Q, Pq, C) after stacking (or list of Q arrays each (Pq, C) before stacking) – per-query LiDAR point clouds in local box-center frame; Pq = QUERY_POINTS_NUM (default 256), C = feature channels
    """Query point sets, shape ``(Q, Pq, C)`` (or list before final stacking)."""
    geo_memory_points: np.ndarray  # shape: (Pm, C) – aggregated track memory point cloud in local box-center frame; Pm = MEMORY_POINTS_NUM (default 4096), C = feature channels (xyz, intensity, p2s_front, p2s_back, score, ...)
    """Track memory points, shape ``(Pm, C)``."""
    geo_trajectory: np.ndarray  # shape: (T, 7+) – sampled track trajectory boxes in world/global frame; columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians; T = number of sampled track frames
    """Track trajectory boxes in global frame, shape ``(T, 7+)``."""
    geo_score: np.ndarray  # shape: (T,) – detection confidence scores for the sampled track frames; float32 in [0, 1]
    gt_geo_query_boxes: np.ndarray  # shape: (Q, 7) – GT boxes for the query proposals in local box-center frame with center and heading zeroed out; same layout as geo_query_boxes; meters/radians
    gt_geo_trajectory: np.ndarray  # shape: (T, 7+) – GT trajectory boxes in world/global frame; same column layout as geo_trajectory; x/y/z/dx/dy/dz in meters, heading in radians
    pose: np.ndarray  # shape: (T, 4, 4) – ego-to-world SE(3) transforms for sampled track frames; dtype float64
    state: str
    matched: np.ndarray  # shape: (T,) – per-frame proposal-to-GT match flags; dtype bool
    matched_tracklet: bool


class PositionSampleDict(TypedDict, total=False):
    """Per-object sample dictionary for position refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray  # shape: (T,) – frame IDs for the sampled (unpadded) track steps; T = actual box count before padding; dtype int-like
    obj_id: TrackId
    obj_cls: int
    pos_trajectory: np.ndarray  # shape: (Q, 7) – trajectory boxes in init-box local frame, padded to QUERY_NUM; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters relative to init-box center, dx/dy/dz in meters, heading in radians relative to init-box heading; Q = QUERY_NUM (default 200), padded rows are zeros
    """Trajectory in init-box local frame, shape ``(Q, 7)`` (padded)."""
    gt_pos_trajectory: np.ndarray  # shape: (Q, 7) – GT trajectory boxes in init-box local frame, padded to QUERY_NUM; same column layout as pos_trajectory; padded rows are zeros
    pos_scores: np.ndarray  # shape: (T,) – detection confidence scores for sampled (unpadded) track steps; float32 in [0, 1]; T = actual box count before padding
    pos_init_box: np.ndarray  # shape: (7,) – the reference/anchor box in world/global frame that defines the init-box coordinate origin; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters, heading in radians
    box_num: int
    padding_mask: np.ndarray  # shape: (Q,) – padding indicator; 0 for valid entries, 1 for padded (zero-filled) entries; Q = QUERY_NUM (default 200)
    """Padding mask, shape ``(Q,)`` where 1 indicates padded entries."""
    pos_query_points: np.ndarray  # shape: (Q, Pq, C) – per-proposal LiDAR point features in init-box local frame; Q = QUERY_NUM (default 200, padded), Pq = QUERY_POINTS_NUM (default 256), C = feature channels (xyz, intensity, p2co offsets, score, class one-hot, ...)
    """Local query points, shape ``(Q, Pq, C)``."""
    pos_memory_points: np.ndarray  # shape: (Q, Pm, C) – per-proposal dense context/trajectory points in init-box local frame; Q = QUERY_NUM (default 200, padded), Pm = MEMORY_POINTS_NUM (default 48), C = feature channels
    """Global/context points, shape ``(Q, Pm, C)``."""
    pose: np.ndarray  # shape: (T, 4, 4) – ego-to-world SE(3) transforms for sampled (unpadded) track steps; dtype float64; T = actual box count before padding
    state: str
    matched: np.ndarray  # shape: (T,) – per-frame proposal-to-GT match flags; dtype bool; T = actual box count before padding
    matched_tracklet: bool


class ConfidenceSampleDict(TypedDict, total=False):
    """Per-object sample dictionary for confidence refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray  # shape: (T,) – frame IDs for the sampled (unpadded) track steps; T = actual box count before padding; dtype int-like
    obj_id: TrackId
    conf_score: np.ndarray  # shape: (Q,) – original detection confidence score per box padded to QUERY_NUM; float32 in [0, 1] for valid entries, -1 sentinel for padded entries; Q = QUERY_NUM (default 200)
    """Original score per box, shape ``(Q,)`` with ``-1`` for padded entries."""
    state: str
    matched_tracklet: bool
    iou: np.ndarray  # shape: (Q,) – IoU supervision target per box, padded to QUERY_NUM; float32 in [0, 1] for valid entries, -1 sentinel for padded entries; Q = QUERY_NUM (default 200)
    """IoU supervision target per box, shape ``(Q,)`` with ``-1`` for padding."""
    box_num: int
    conf_points: np.ndarray  # shape: (Q, Pq, C) – per-proposal LiDAR point features in init-box local frame; Q = QUERY_NUM (default 200, padded), Pq = QUERY_POINTS_NUM (default 256), C = feature channels (xyz, intensity, p2co offsets, box_pos, score, ...); padded entries are zero-filled
    """Input points/features, shape ``(Q, Pq, C)``."""


class RefineBatchDict(TypedDict, total=False):
    """Collated batch dictionary used by refining models after ``collate_batch``."""

    batch_size: int
    sequence_name: np.ndarray  # shape: (B,) – sequence name string for each batch element; dtype object (str); B = batch size
    frame: np.ndarray  # shape: (B, T) – frame IDs per batch element; T = variable track length or QUERY_NUM depending on the task; dtype int-like
    obj_id: np.ndarray  # shape: (B,) – object/track ID for each batch element; dtype int or str
    obj_cls: NumpyOrTorchArray  # shape: (B,) – integer class index for each batch element; 1 = Vehicle, 2 = Pedestrian, 3 = Cyclist
    pose: np.ndarray  # shape: (B, T, 4, 4) – ego-to-world SE(3) transforms per frame per batch element; dtype float64
    state: List[str]
    matched: NumpyOrTorchArray  # shape: (B, T) – per-frame proposal-to-GT match flags; dtype bool; T = track length
    matched_tracklet: List[bool]
    geo_query_num: NumpyOrTorchArray  # shape: (B,) – number of valid (unpadded) query proposals per batch element; dtype int
    geo_query_boxes: NumpyOrTorchArray  # shape: (B, Q, 7) – query boxes in local box-center frame; columns: [0, 0, 0, dx, dy, dz, 0] (center and heading zeroed); dx/dy/dz in meters; Q = max geo_query_num in batch (padded with zeros)
    geo_query_points: NumpyOrTorchArray  # shape: (B, Q, Pq, C) – per-query LiDAR point clouds in local box-center frame; Q = max geo_query_num in batch, Pq = QUERY_POINTS_NUM (default 256), C = feature channels
    geo_memory_points: NumpyOrTorchArray  # shape: (B, Pm, C) – aggregated track memory point cloud in local box-center frame; Pm = MEMORY_POINTS_NUM (default 4096), C = feature channels
    geo_trajectory: np.ndarray  # shape: (B, T, 7+) – sampled track trajectory boxes in world/global frame; columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians
    geo_score: NumpyOrTorchArray  # shape: (B, T) – detection confidence scores for sampled track frames; float32 in [0, 1]
    gt_geo_query_boxes: NumpyOrTorchArray  # shape: (B, Q, 7) – GT query boxes in local box-center frame; same column layout as geo_query_boxes; padded with zeros to Q
    gt_geo_trajectory: NumpyOrTorchArray  # shape: (B, T, 7+) – GT trajectory boxes in world/global frame; same column layout as geo_trajectory
    pos_trajectory: NumpyOrTorchArray  # shape: (B, Q, 7) – trajectory boxes in init-box local frame, padded to QUERY_NUM; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters relative to init-box, heading in radians relative to init-box; Q = QUERY_NUM (default 200)
    gt_pos_trajectory: NumpyOrTorchArray  # shape: (B, Q, 7) – GT trajectory boxes in init-box local frame, padded to QUERY_NUM; same column layout as pos_trajectory
    pos_scores: NumpyOrTorchArray  # shape: (B, T) – detection confidence scores for sampled (unpadded) position track steps; float32 in [0, 1]; T = actual box count before padding
    pos_init_box: NumpyOrTorchArray  # shape: (B, 7) – reference anchor box in world/global frame defining the init-box coordinate origin; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters, heading in radians
    padding_mask: NumpyOrTorchArray  # shape: (B, Q) – padding indicator for position/confidence; 0 = valid entry, 1 = padded (zero-filled); Q = QUERY_NUM (default 200)
    pos_query_points: NumpyOrTorchArray  # shape: (B, Q, Pq, C) – per-proposal LiDAR point features in init-box local frame; Q = QUERY_NUM (default 200, padded), Pq = QUERY_POINTS_NUM (default 256), C = feature channels
    pos_memory_points: NumpyOrTorchArray  # shape: (B, Q, Pm, C) – per-proposal dense context points in init-box local frame; Q = QUERY_NUM (default 200, padded), Pm = MEMORY_POINTS_NUM (default 48), C = feature channels
    iou: NumpyOrTorchArray  # shape: (B, Q) – IoU supervision targets, padded to QUERY_NUM; float32 in [0, 1] for valid entries, -1 sentinel for padded entries; Q = QUERY_NUM (default 200)
    box_num: NumpyOrTorchArray  # shape: (B,) – number of valid (unpadded) boxes per batch element; dtype int
    conf_score: NumpyOrTorchArray  # shape: (B, Q) – original detection confidence scores, padded to QUERY_NUM; float32 in [0, 1] for valid, -1 for padded; Q = QUERY_NUM (default 200)
    conf_points: NumpyOrTorchArray  # shape: (B, Q, Pq, C) – per-proposal LiDAR point features for confidence model; Q = QUERY_NUM (default 200, padded), Pq = QUERY_POINTS_NUM (default 256), C = feature channels
    query: torch.Tensor  # shape: (B, D, Q) – encoded query features from query encoder/MLP; D = embed_dims (default 256), Q = query count (geo: Q proposals; pos: QUERY_NUM); populated inside forward() of GeometryTransformer/PositionTransformer
    memory: torch.Tensor  # shape: (B, D, L) – encoded memory (key/value) features from memory encoder/MLP; D = embed_dims (default 256), L = memory sequence length (geo: Pm; pos: Q*Pm); populated inside forward()
    query_pos: torch.Tensor  # shape: (B, Q, 4) – positional encoding for queries derived from trajectory; columns: [x, y, z, heading]; x/y/z in meters (init-box local frame), heading in radians; used only by PositionTransformer
    batch_box_preds: torch.Tensor  # shape: (B, 7) for geometry or (B, Q, 7) for position – predicted bounding box parameters; geometry: single refined box per track [x, y, z, dx, dy, dz, heading]; position: per-proposal boxes; x/y/z/dx/dy/dz in meters, heading in radians
    pred_score: torch.Tensor  # shape: (B, Q) – predicted confidence scores after confidence refining; float32 in [0, 1]; Q = QUERY_NUM (default 200); padded positions correspond to padding_mask == 1


class GeometryBatchDict(RefineBatchDict, total=False):
    """Geometry-refining batch view (keys relevant to geometry flow)."""


class PositionBatchDict(RefineBatchDict, total=False):
    """Position-refining batch view (keys relevant to position flow)."""


class ConfidenceBatchDict(RefineBatchDict, total=False):
    """Confidence-refining batch view (keys relevant to confidence flow)."""


class GeometryPredDict(TypedDict):
    """Model prediction dictionary returned by geometry refining model."""

    pred_boxes: np.ndarray  # shape: (B, 7) – refined geometry box per batch element; columns: [x, y, z, dx, dy, dz, heading]; x/y/z/dx/dy/dz in meters, heading in radians; produced by GeometryTransformer.generate_predicted_boxes(); averaged across all query proposals and decoder layers
    pose: np.ndarray  # shape: (B, T, 4, 4) – ego-to-world SE(3) transforms per batch element and frame step; dtype float64; copied from batch_dict['pose']
    geo_trajectory: np.ndarray  # shape: (B, T, 7+) – track trajectory boxes in world/global frame; columns: [x, y, z, dx, dy, dz, heading, ...]; x/y/z/dx/dy/dz in meters, heading in radians; copied from batch_dict['geo_trajectory']


class PositionPredDict(TypedDict):
    """Model prediction dictionary returned by position refining model."""

    pred_boxes: np.ndarray  # shape: (B, Q, 7) – refined per-proposal boxes in init-box local frame; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters, dx/dy/dz in meters, heading in radians; Q = QUERY_NUM (default 200, includes padding)
    pose: np.ndarray  # shape: (B, T, 4, 4) – ego-to-world SE(3) transforms per batch element and unpadded frame step; dtype float64; T = actual box count (unpadded)
    pos_init_box: np.ndarray  # shape: (B, 7) – reference anchor box in world/global frame; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters, heading in radians; used to revert init-box local coords back to global frame
    gt_pos_trajectory: np.ndarray  # shape: (B, Q, 7) – GT trajectory boxes in init-box local frame, padded to QUERY_NUM; columns: [x, y, z, dx, dy, dz, heading]; x/y/z in meters, heading in radians


class ConfidencePredDict(TypedDict):
    """Model prediction dictionary returned by confidence refining model."""

    pred_score: np.ndarray  # shape: (B, Q) – predicted confidence scores after confidence refining; float32 in [0, 1]; Q = QUERY_NUM (default 200); padded positions (padding_mask == 1) should be ignored


class GeometryTrackPrediction(TypedDict):
    """Saved per-track geometry prediction results."""

    sequence_name: SequenceName
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]  # list of T elements; each element shape (1, 7) – predicted refined box in ego-vehicle/LiDAR frame for each frame; columns: [x, y, z, dx, dy, dz, heading]; x/y/z/dx/dy/dz in meters, heading in radians; T = number of frames in this track
    score: List[float]
    name: List[str]
    pose: List[np.ndarray]  # list of T elements; each element shape (4, 4) – ego-to-world SE(3) transform for the corresponding frame; dtype float64


class PositionTrackPrediction(TypedDict):
    """Saved per-track position prediction results."""

    sequence_name: SequenceName
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]  # list of T elements; each element shape (7,) – predicted refined box in ego-vehicle/LiDAR frame; columns: [x, y, z, dx, dy, dz, heading]; x/y/z/dx/dy/dz in meters, heading in radians; T = number of frames in this track
    boxes_global: List[np.ndarray]  # list of T elements; each element shape (7,) – predicted refined box in world/global frame; same column layout as boxes_lidar; x/y/z/dx/dy/dz in meters, heading in radians
    score: List[float]
    name: List[str]
    state: str
    pose: List[np.ndarray]  # list of T elements; each element shape (4, 4) – ego-to-world SE(3) transform for the corresponding frame; dtype float64
    boxes_gt: List[np.ndarray]  # list of T elements; each element shape (7,) – GT box in ego-vehicle/LiDAR frame for each frame; columns: [x, y, z, dx, dy, dz, heading]; meters/radians
    boxes_gt_global: List[np.ndarray]  # list of T elements; each element shape (7,) – GT box in world/global frame; same column layout as boxes_gt; meters/radians


class ConfidenceTrackPrediction(TypedDict):
    """Saved per-track confidence prediction results."""

    sequence_name: SequenceName
    frame_id: np.ndarray  # shape: (T,) – frame IDs for each step of this track; T = actual box count (unpadded); dtype int
    score: np.ndarray  # shape: (T,) – original detection confidence scores; float32 in [0, 1]; T = actual box count (unpadded)
    new_score: np.ndarray  # shape: (T,) – refined confidence scores output by the confidence model; float32 in [0, 1]; T = actual box count (unpadded)


GeometryPredictionStore = Dict[SequenceName, Dict[TrackId, GeometryTrackPrediction]]
PositionPredictionStore = Dict[SequenceName, Dict[TrackId, PositionTrackPrediction]]
ConfidencePredictionStore = Dict[SequenceName, Dict[TrackId, ConfidenceTrackPrediction]]
