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

ArrayLike = Union[np.ndarray, torch.Tensor]
TrackId = Union[int, str]


class TrackObjectInfo(TypedDict, total=False):
    """Object-track data loaded from refining input pickle files.

    ``T`` denotes the track length (number of frames for this object track).
    """

    sequence_name: SequenceName
    obj_id: TrackId
    name: str
    sample_idx: np.ndarray
    """Frame IDs for this object track, shape ``(T,)``."""
    boxes_global: np.ndarray
    """Track boxes in world frame, shape ``(T, 7+)``: ``[x, y, z, dx, dy, dz, heading, ...]``."""
    gt_boxes_global: np.ndarray
    """Matched GT boxes in world frame, shape ``(T, 7+)``."""
    score: np.ndarray
    """Detection scores per frame, shape ``(T,)``."""
    pose: np.ndarray
    """Ego-to-world transforms, shape ``(T, 4, 4)``."""
    pts: List[np.ndarray]
    """Per-frame cropped point clouds, list length ``T``, each shape ``(Pi, C)``."""
    matched: np.ndarray
    """Per-frame proposal-to-GT match flag, shape ``(T,)`` bool."""
    matched_tracklet: bool
    state: str
    """Track motion state (e.g. ``'static'`` or ``'dynamic'``)."""
    refine_iou: np.ndarray
    """Per-frame IoU target used by confidence refining, shape ``(T,)``."""


class GeometrySampleDict(TypedDict, total=False):
    """Per-object sample dictionary for geometry refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray
    obj_id: TrackId
    obj_cls: int
    geo_query_num: int
    geo_query_boxes: np.ndarray
    """Query boxes in local normalized frame, shape ``(Q, 7)``."""
    geo_query_points: Union[np.ndarray, List[np.ndarray]]
    """Query point sets, shape ``(Q, Pq, C)`` (or list before final stacking)."""
    geo_memory_points: np.ndarray
    """Track memory points, shape ``(Pm, C)``."""
    geo_trajectory: np.ndarray
    """Track trajectory boxes in global frame, shape ``(T, 7+)``."""
    geo_score: np.ndarray
    gt_geo_query_boxes: np.ndarray
    gt_geo_trajectory: np.ndarray
    pose: np.ndarray
    state: str
    matched: np.ndarray
    matched_tracklet: bool


class PositionSampleDict(TypedDict, total=False):
    """Per-object sample dictionary for position refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray
    obj_id: TrackId
    obj_cls: int
    pos_trajectory: np.ndarray
    """Trajectory in init-box local frame, shape ``(Q, 7)`` (padded)."""
    gt_pos_trajectory: np.ndarray
    pos_scores: np.ndarray
    pos_init_box: np.ndarray
    box_num: int
    padding_mask: np.ndarray
    """Padding mask, shape ``(Q,)`` where 1 indicates padded entries."""
    pos_query_points: np.ndarray
    """Local query points, shape ``(Q, Pq, C)``."""
    pos_memory_points: np.ndarray
    """Global/context points, shape ``(Q, Pm, C)``."""
    pose: np.ndarray
    state: str
    matched: np.ndarray
    matched_tracklet: bool


class ConfidenceSampleDict(TypedDict, total=False):
    """Per-object sample dictionary for confidence refining before collation."""

    sequence_name: SequenceName
    frame: np.ndarray
    obj_id: TrackId
    conf_score: np.ndarray
    """Original score per box, shape ``(Q,)`` with ``-1`` for padded entries."""
    state: str
    matched_tracklet: bool
    iou: np.ndarray
    """IoU supervision target per box, shape ``(Q,)`` with ``-1`` for padding."""
    box_num: int
    conf_points: np.ndarray
    """Input points/features, shape ``(Q, Pq, C)``."""


class RefineBatchDict(TypedDict, total=False):
    """Collated batch dictionary used by refining models after ``collate_batch``."""

    batch_size: int
    sequence_name: np.ndarray
    frame: np.ndarray
    obj_id: np.ndarray
    obj_cls: ArrayLike
    pose: np.ndarray
    state: List[str]
    matched: ArrayLike
    matched_tracklet: List[bool]
    geo_query_num: ArrayLike
    geo_query_boxes: ArrayLike
    geo_query_points: ArrayLike
    geo_memory_points: ArrayLike
    geo_trajectory: np.ndarray
    geo_score: ArrayLike
    gt_geo_query_boxes: ArrayLike
    gt_geo_trajectory: ArrayLike
    pos_trajectory: ArrayLike
    gt_pos_trajectory: ArrayLike
    pos_scores: ArrayLike
    pos_init_box: ArrayLike
    padding_mask: ArrayLike
    pos_query_points: ArrayLike
    pos_memory_points: ArrayLike
    iou: ArrayLike
    box_num: ArrayLike
    conf_score: ArrayLike
    conf_points: ArrayLike
    query: torch.Tensor
    memory: torch.Tensor
    query_pos: torch.Tensor
    batch_box_preds: torch.Tensor
    pred_score: torch.Tensor


GeometryBatchDict = RefineBatchDict
PositionBatchDict = RefineBatchDict
ConfidenceBatchDict = RefineBatchDict


class GeometryPredDict(TypedDict):
    """Model prediction dictionary returned by geometry refining model."""

    pred_boxes: np.ndarray
    pose: np.ndarray
    geo_trajectory: np.ndarray


class PositionPredDict(TypedDict):
    """Model prediction dictionary returned by position refining model."""

    pred_boxes: np.ndarray
    pose: np.ndarray
    pos_init_box: np.ndarray
    gt_pos_trajectory: np.ndarray


class ConfidencePredDict(TypedDict):
    """Model prediction dictionary returned by confidence refining model."""

    pred_score: np.ndarray


class GeometryTrackPrediction(TypedDict):
    """Saved per-track geometry prediction results."""

    sequence_name: SequenceName
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]
    score: List[float]
    name: List[str]
    pose: List[np.ndarray]


class PositionTrackPrediction(TypedDict):
    """Saved per-track position prediction results."""

    sequence_name: SequenceName
    frame_id: List[int]
    boxes_lidar: List[np.ndarray]
    boxes_global: List[np.ndarray]
    score: List[float]
    name: List[str]
    state: str
    pose: List[np.ndarray]
    boxes_gt: List[np.ndarray]
    boxes_gt_global: List[np.ndarray]


class ConfidenceTrackPrediction(TypedDict):
    """Saved per-track confidence prediction results."""

    sequence_name: SequenceName
    frame_id: np.ndarray
    score: np.ndarray
    new_score: np.ndarray


GeometryPredictionStore = Dict[SequenceName, Dict[TrackId, GeometryTrackPrediction]]
PositionPredictionStore = Dict[SequenceName, Dict[TrackId, PositionTrackPrediction]]
ConfidencePredictionStore = Dict[SequenceName, Dict[TrackId, ConfidenceTrackPrediction]]
