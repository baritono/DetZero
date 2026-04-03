"""
Typed schema definitions for the detection data pipeline.

These types document the structure, shapes, dtypes, and semantic
meanings of the dictionaries that flow through the detection pipeline —
from raw dataset loading through voxelisation to model input and output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from typing import TypedDict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# InputDict
# ---------------------------------------------------------------------------

class InputDict(TypedDict, total=False):
    """
    Raw input produced by ``DatasetTemplate.__getitem__`` before augmentation
    and voxelisation.

    Fields
    ------
    points : np.ndarray, shape (N, C_in), dtype float32
        Point cloud.  Each row is one point with features
        ``[x, y, z, intensity, elongation]`` in the current LIDAR frame.
        When multi-sweep fusion is enabled a 6th column
        ``time_offset`` (seconds, float32) is appended.
        ``C_in`` is typically 5 (single sweep) or 6 (multi-sweep).
    frame_id : int
        Zero-based sample index within the parent sequence.
    pose : np.ndarray, shape (4, 4), dtype float64
        Rigid-body transformation from the LIDAR frame of this sample to the
        global (world) coordinate system.
    sequence_name : str
        Human-readable identifier of the driving sequence this sample belongs
        to (e.g. ``'segment-…'``).
    gt_names : np.ndarray, shape (M,), dtype object (str)
        Class name of each ground-truth object.
        Present only during training.
        Valid values: ``'Vehicle'``, ``'Pedestrian'``, ``'Cyclist'``.
    gt_boxes : np.ndarray, shape (M, 7), dtype float32
        Ground-truth 3-D bounding boxes in the LIDAR frame.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
        ``(x, y, z)`` is the box centre; ``(dx, dy, dz)`` are full extents;
        ``yaw`` is the heading angle in radians (anti-clockwise from x-axis).
        Present only during training.
    gt_boxes_mask : np.ndarray, shape (M,), dtype bool
        ``True`` for boxes whose class is in the target class list.
        Populated by :meth:`DatasetTemplate.prepare_data` before passing to
        the augmentor.  Present only during training.
    """

    points: np.ndarray          # (N, C_in)  float32
    frame_id: int
    pose: np.ndarray            # (4, 4)     float64
    sequence_name: str
    gt_names: np.ndarray        # (M,)       str
    gt_boxes: np.ndarray        # (M, 7)     float32
    gt_boxes_mask: np.ndarray   # (M,)       bool


# ---------------------------------------------------------------------------
# DataDict
# ---------------------------------------------------------------------------

class DataDict(TypedDict, total=False):
    """
    Processed single-sample dictionary returned by
    ``DatasetTemplate.__getitem__``.

    This dict is created from :class:`InputDict`, then passed through the
    augmentor, point-feature encoder, and data-processor in sequence.  All
    fields from :class:`InputDict` that survive those stages are present, plus
    the new fields listed below.

    Fields (new / modified relative to InputDict)
    ----------------------------------------------
    points : np.ndarray, shape (N, C_out), dtype float32
        Point features after encoding.  ``C_out`` equals
        ``PointFeatureEncoder.num_point_features`` (typically 5 or 6).
    gt_boxes : np.ndarray, shape (M, 8), dtype float32
        Boxes extended with a class index in the last column:
        ``[x, y, z, dx, dy, dz, yaw, class_id]``.
        ``class_id`` is 1-based (1 = Vehicle, 2 = Pedestrian, 3 = Cyclist).
        ``gt_names`` is removed from the dict before the item is returned.
    use_lead_xyz : bool
        ``True`` when ``(x, y, z)`` coordinates are used as point-wise
        features in the VFE (set by :class:`PointFeatureEncoder`).
    voxels : np.ndarray, shape (V, P, C_out), dtype float32
        Voxelised point cloud.
        ``V`` = number of non-empty voxels (≤ ``MAX_NUMBER_OF_VOXELS``).
        ``P`` = ``MAX_POINTS_PER_VOXEL`` (typically 5).
        ``C_out`` = point-feature dimension (same as ``points`` above).
    voxel_coords : np.ndarray, shape (V, 3), dtype int32
        Integer voxel grid indices ``[z_idx, y_idx, x_idx]`` for each voxel.
        Note: the batch index is *not* included here; it is added during
        :meth:`DatasetTemplate.collate_batch`.
    voxel_num_points : np.ndarray, shape (V,), dtype int32
        Number of points occupying each voxel (≤ ``P``).
    """

    points: np.ndarray          # (N, C_out)     float32
    frame_id: int
    pose: np.ndarray            # (4, 4)         float64
    sequence_name: str
    gt_boxes: np.ndarray        # (M, 8)         float32  — only during training
    use_lead_xyz: bool
    voxels: np.ndarray          # (V, P, C_out)  float32
    voxel_coords: np.ndarray    # (V, 3)         int32
    voxel_num_points: np.ndarray  # (V,)         int32


# ---------------------------------------------------------------------------
# BatchDict
# ---------------------------------------------------------------------------

class BatchDict(TypedDict, total=False):
    """
    Collated batch dictionary produced by :meth:`DatasetTemplate.collate_batch`
    and fed into the model's ``forward`` method.

    Voxel arrays are concatenated across the batch (total ``V_total`` voxels),
    with a batch index prepended to ``voxel_coords``.  Metadata arrays are
    stacked along a new batch axis.

    Fields
    ------
    voxels : np.ndarray, shape (V_total, P, C_out), dtype float32
        All voxels from all samples in the batch, concatenated.
    voxel_coords : np.ndarray, shape (V_total, 4), dtype int32
        Voxel indices with batch index prepended:
        ``[batch_idx, z_idx, y_idx, x_idx]``.
    voxel_num_points : np.ndarray, shape (V_total,), dtype int32
        Number of points per voxel across the whole batch.
    points : np.ndarray, shape (N_total, 1+C_out), dtype float32
        (Optional) Raw points with batch index prepended:
        ``[batch_idx, x, y, z, ...]``.
        Present when ``transform_points_to_voxels`` is not used (raw-point
        pipeline).
    frame_id : np.ndarray, shape (B,), dtype object (int)
        Sample indices for each element in the batch.
    pose : np.ndarray, shape (B, 4, 4), dtype float64
        Per-sample LIDAR-to-world transformation matrices.
    sequence_name : np.ndarray, shape (B,), dtype object (str)
        Sequence names for each element in the batch.
    batch_size : int
        Number of samples in the batch (``B``).
        When TTA is active this equals ``B * num_tta_variants``.
    gt_boxes : np.ndarray, shape (B, M_max, 8), dtype float32
        Zero-padded ground-truth boxes.  ``M_max`` is the maximum number of
        objects in any sample of the batch.  Rows beyond the actual object
        count for a given sample are filled with zeros.
        Present only during training.
    use_lead_xyz : bool
        Forwarded from the per-sample :class:`DataDict`.
    tta_ops : List[str]
        Names of the TTA variants applied (e.g. ``'tta_original'``,
        ``'tta_flip_x'``).  Present only when TTA is enabled.

    Model-populated fields (added in-place by model sub-modules)
    ------------------------------------------------------------
    spatial_features : torch.Tensor, shape (B, C, H, W), dtype float32
        BEV feature map output of the 3-D backbone / map-to-BEV module.
    spatial_features_2d : torch.Tensor, shape (B, C, H, W), dtype float32
        BEV feature map output of the 2-D backbone.
    spatial_features_2d_strides : int or None
        Spatial stride of ``spatial_features_2d`` relative to input grid.
    batch_cls_preds : torch.Tensor, shape (B, num_rois, num_classes) or (B, num_rois, 1), dtype float32
        Class (or IoU) predictions from the first-stage or ROI head.
    batch_box_preds : torch.Tensor, shape (B, num_rois, 7), dtype float32
        Box predictions from the first-stage or ROI head.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
    cls_preds_normalized : bool
        Whether ``batch_cls_preds`` values are already sigmoid-normalised.
    rois : torch.Tensor, shape (B, num_rois, 7), dtype float32
        Region-of-interest proposals from the first stage (second-stage
        pipeline only).
    roi_scores : torch.Tensor, shape (B, num_rois), dtype float32
        Proposal confidence scores.
    roi_labels : torch.Tensor, shape (B, num_rois), dtype int64
        Proposal class labels (1-based).
    roi_features : torch.Tensor, shape (B, num_rois, C_feat), dtype float32
        Pooled BEV features for each ROI.
    has_class_labels : bool
        ``True`` when ``roi_labels`` contains valid class information.
    final_box_dicts : List[PredDict]
        First-stage decoded predictions (one per sample), set by the dense
        head when *not* in second-stage mode.
    multihead_label_mapping : List[torch.Tensor]
        Mapping from per-head class indices to global class indices; used in
        multi-class NMS.
    """

    voxels: np.ndarray              # (V_total, P, C_out)      float32
    voxel_coords: np.ndarray        # (V_total, 4)             int32
    voxel_num_points: np.ndarray    # (V_total,)               int32
    points: np.ndarray              # (N_total, 1+C_out)       float32
    frame_id: np.ndarray            # (B,)                     int/object
    pose: np.ndarray                # (B, 4, 4)                float64
    sequence_name: np.ndarray       # (B,)                     str
    batch_size: int
    gt_boxes: np.ndarray            # (B, M_max, 8)            float32
    use_lead_xyz: bool
    tta_ops: List[str]
    # --- model-populated ---
    spatial_features: torch.Tensor          # (B, C, H, W)
    spatial_features_2d: torch.Tensor       # (B, C, H, W)
    spatial_features_2d_strides: int
    batch_cls_preds: torch.Tensor           # (B, num_rois, num_cls|1)
    batch_box_preds: torch.Tensor           # (B, num_rois, 7)
    cls_preds_normalized: bool
    rois: torch.Tensor                      # (B, num_rois, 7)
    roi_scores: torch.Tensor                # (B, num_rois)
    roi_labels: torch.Tensor                # (B, num_rois)  int64
    roi_features: torch.Tensor              # (B, num_rois, C_feat)
    has_class_labels: bool
    final_box_dicts: List[PredDict]
    multihead_label_mapping: List[torch.Tensor]


# ---------------------------------------------------------------------------
# PredDict
# ---------------------------------------------------------------------------

@dataclass
class PredDict:
    """
    Per-sample prediction result returned by the detection model at inference.

    Produced by :meth:`CenterHead.generate_predicted_boxes` (first-stage) or
    the second-stage ROI head, and gathered in
    :meth:`CenterPoint.post_processing`.

    Attribute access is preferred (``pred.pred_boxes``).  Dict-style access
    (``pred['pred_boxes']``) is also supported for backward compatibility.

    Attributes
    ----------
    pred_boxes : torch.Tensor, shape (N, 7), dtype float32
        Detected 3-D bounding boxes in the LIDAR frame.
        Each row: ``[x, y, z, dx, dy, dz, yaw]``.
        ``N`` is the number of detections after NMS.
    pred_scores : torch.Tensor, shape (N,), dtype float32
        Detection confidence score in ``[0, 1]`` for each box.
        In the second-stage pipeline this is the geometric mean of the IoU
        prediction and the first-stage score.
    pred_labels : torch.Tensor, shape (N,), dtype int64
        1-based class index.
        ``1`` = Vehicle, ``2`` = Pedestrian, ``3`` = Cyclist.
    """

    pred_boxes: torch.Tensor    # (N, 7)  float32  — [x, y, z, dx, dy, dz, yaw]
    pred_scores: torch.Tensor   # (N,)    float32  — confidence in [0, 1]
    pred_labels: torch.Tensor   # (N,)    int64    — 1-based class index

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        setattr(self, key, value)

    def __len__(self) -> int:
        return len(self.pred_boxes)



# ---------------------------------------------------------------------------
# AnnoDictEntry
# ---------------------------------------------------------------------------

class AnnoDictEntry(TypedDict):
    """
    Per-sample annotation entry produced by
    :meth:`DatasetTemplate.generate_prediction_dicts`.

    Suitable for serialisation and downstream evaluation.

    Fields
    ------
    name : np.ndarray, shape (N,), dtype object (str)
        Class name string for each detected object.
    score : np.ndarray, shape (N,), dtype float32
        Detection confidence score for each box.
    boxes_lidar : np.ndarray, shape (N, 7) or (N, 9), dtype float32
        Bounding boxes in the LIDAR frame.  The base 7 columns are
        ``[x, y, z, dx, dy, dz, yaw]``; columns 8-9 are
        ``[vel_x, vel_y]`` when velocity prediction is enabled.
    sequence_name : str
        Sequence identifier forwarded from the batch.
    frame_id : int
        Sample index within the sequence.
    pose : np.ndarray, shape (4, 4), dtype float64
        LIDAR-to-world transformation for this frame.
    """

    name: np.ndarray        # (N,)      str
    score: np.ndarray       # (N,)      float32
    boxes_lidar: np.ndarray  # (N, 7|9) float32
    sequence_name: str
    frame_id: int
    pose: np.ndarray        # (4, 4)   float64
