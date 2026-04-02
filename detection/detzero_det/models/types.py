"""
Typed schema definitions for detection model inputs and outputs.

These TypedDict / NamedTuple classes document the structures that flow
between the detection model's sub-modules (VFE → Backbone3D → MapToBEV →
Backbone2D → CenterHead / ROI Head) and the training / inference entry
points.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple
from typing import TypedDict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# HeadPredDict
# ---------------------------------------------------------------------------

class HeadPredDict(TypedDict, total=False):
    """
    Raw per-class-group prediction tensors produced by
    :class:`SeparateHead` (one dict per detection head).

    All tensors have a leading batch dimension ``B`` and spatial dimensions
    ``H × W`` corresponding to the BEV feature map.

    Fields
    ------
    hm : torch.Tensor, shape (B, C_head, H, W), dtype float32
        Class heatmaps (logits before sigmoid).
        ``C_head`` = number of classes assigned to this head group.
    center : torch.Tensor, shape (B, 2, H, W), dtype float32
        Sub-voxel xy centre offset from the grid-cell centre.
        Values are in voxel units; typical range ≈ ``[-0.5, 0.5]``.
    center_z : torch.Tensor, shape (B, 1, H, W), dtype float32
        Absolute z coordinate of the box centre (metres).
    dim : torch.Tensor, shape (B, 3, H, W), dtype float32
        Log-space box dimensions: ``log(dx)``, ``log(dy)``, ``log(dz)``.
        Apply ``exp()`` to recover metres.
    rot : torch.Tensor, shape (B, 2, H, W), dtype float32
        Rotation encoded as ``[sin(yaw), cos(yaw)]``.
    iou : torch.Tensor, shape (B, 1, H, W), dtype float32
        IoU prediction head output (logits).  Present only when
        ``IOU_WEIGHT > 0`` in the model config.
    vel : torch.Tensor, shape (B, 2, H, W), dtype float32
        Velocity prediction ``[vel_x, vel_y]`` in m/s.  Present only when
        ``'vel'`` is in ``HEAD_ORDER`` (e.g. for nuScenes-style configs).
    """

    hm: torch.Tensor        # (B, C_head, H, W)
    center: torch.Tensor    # (B, 2, H, W)
    center_z: torch.Tensor  # (B, 1, H, W)
    dim: torch.Tensor       # (B, 3, H, W)
    rot: torch.Tensor       # (B, 2, H, W)
    iou: torch.Tensor       # (B, 1, H, W)   — optional
    vel: torch.Tensor       # (B, 2, H, W)   — optional


# ---------------------------------------------------------------------------
# TargetDict
# ---------------------------------------------------------------------------

class TargetDict(TypedDict):
    """
    Ground-truth target tensors produced by
    :meth:`CenterHead.assign_targets`.

    Each list entry corresponds to one detection head group.

    Fields
    ------
    heatmaps : List[torch.Tensor]
        Per-head Gaussian heatmaps, each shape
        ``(B, C_head, H, W)``, dtype float32.
    target_boxes : List[torch.Tensor]
        Per-head regression targets at each foreground location,
        each shape ``(B, num_max_objs, 8)``, dtype float32.
        Columns: ``[x_offset, y_offset, z, log_dx, log_dy, log_dz,
        sin_yaw, cos_yaw]``.
    inds : List[torch.Tensor]
        Flattened spatial indices of foreground locations,
        each shape ``(B, num_max_objs)``, dtype int64.
    masks : List[torch.Tensor]
        Binary validity mask for ``target_boxes`` and ``inds``,
        each shape ``(B, num_max_objs)``, dtype bool.
    heatmap_masks : List[torch.Tensor]
        Per-head mask used for heatmap loss weighting,
        each shape ``(B, C_head, H, W)``, dtype bool.
    """

    heatmaps: List[torch.Tensor]
    target_boxes: List[torch.Tensor]
    inds: List[torch.Tensor]
    masks: List[torch.Tensor]
    heatmap_masks: List[torch.Tensor]


# ---------------------------------------------------------------------------
# TrainingOutput
# ---------------------------------------------------------------------------

class TrainingOutput(NamedTuple):
    """
    Return value of :meth:`CenterPoint.forward` during training.

    Fields
    ------
    loss_dict : Dict[str, torch.Tensor]
        Scalar loss tensors keyed by name.  Always contains ``'loss'``
        (total loss).  Other keys are added when a second-stage head is
        active (e.g. ``'loss_rcnn'``).
    tb_dict : Dict[str, float]
        Scalar metrics logged to TensorBoard (e.g. per-head heatmap loss,
        regression loss, IoU loss).
    disp_dict : Dict[str, float]
        Lightweight subset of metrics displayed on the console / progress
        bar (currently empty; reserved for future use).
    """

    loss_dict: Dict[str, torch.Tensor]
    tb_dict: Dict[str, float]
    disp_dict: Dict[str, float]


# ---------------------------------------------------------------------------
# RecallDict
# ---------------------------------------------------------------------------

class RecallDict(TypedDict, total=False):
    """
    Accumulator for recall metrics computed during inference by
    :meth:`CenterPoint.generate_recall_record`.

    Fields
    ------
    gt : int
        Total number of ground-truth objects seen so far.
    roi_<thresh> : int
        Number of GT objects recalled at IoU ≥ ``<thresh>`` by the first-
        stage ROI proposals.  One key per threshold in
        ``POST_PROCESSING.RECALL_THRESH_LIST``.
    rcnn_<thresh> : int
        Number of GT objects recalled at IoU ≥ ``<thresh>`` by the final
        (optionally second-stage refined) predictions.
    """

    gt: int
    # Keys like 'roi_0.3', 'rcnn_0.5', … are added dynamically at runtime.


# ---------------------------------------------------------------------------
# ModelInfoDict
# ---------------------------------------------------------------------------

class ModelInfoDict(TypedDict, total=False):
    """
    Intermediate configuration dict populated during
    :meth:`CenterPoint.build_networks` and passed between module
    constructors.

    Fields
    ------
    module_list : List
        Accumulated list of instantiated sub-modules.
    num_point_features : int
        Number of point-feature channels at the current stage of the
        pipeline (updated after VFE, Backbone3D, etc.).
    num_bev_features : int
        Number of BEV feature channels at the current stage.
    grid_size : np.ndarray, shape (3,), dtype int64
        ``[W, H, D]`` voxel grid extents.
    point_cloud_range : np.ndarray, shape (6,), dtype float32
        ``[x_min, y_min, z_min, x_max, y_max, z_max]`` in metres.
    voxel_size : List[float]
        ``[dx, dy, dz]`` voxel dimensions in metres.
    """

    module_list: List
    num_point_features: int
    num_bev_features: int
    grid_size: np.ndarray   # (3,)  int64
    point_cloud_range: np.ndarray  # (6,)  float32
    voxel_size: List[float]
