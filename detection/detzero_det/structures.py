"""
Typed data-structure definitions for the DetZero detection pipeline.

Every dictionary that flows between pipeline components is defined here as a
:class:`typing.TypedDict` so that:

* the field names are explicit and discoverable,
* the tensor/array shapes and dtypes are documented in the docstring of each
  field,
* static type-checkers (mypy, pyright) can verify correctness, and
* IDEs can provide auto-complete for field names.

Naming conventions
------------------
- Fields that exist in more than one TypedDict and carry the same semantic
  meaning use **exactly the same field name** (e.g. ``gt_boxes``, ``rois``,
  ``roi_scores``, ``roi_labels``, ``pred_boxes``, ``pred_scores``,
  ``pred_labels``).
- Shape annotations use abbreviated dimension names:
    ``B`` – batch size,
    ``N`` – number of points / voxels / proposals (context-dependent),
    ``M`` – number of GT boxes per frame,
    ``H``, ``W`` – spatial height / width of a feature map,
    ``C`` – number of feature channels,
    ``D`` – depth dimension of a 3-D feature tensor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from typing_extensions import TypedDict

# AnnotationDict is the detection→tracking interface; it is defined in the
# shared detzero_utils package so that the tracking module can import it
# without taking a dependency on detzero_det.
from detzero_utils.structures import AnnotationDict, FrameId  # noqa: F401  (re-exported)


# ---------------------------------------------------------------------------
# Nested helper dicts
# ---------------------------------------------------------------------------

class AugMatrixInv(TypedDict, total=False):
    """Inverse-transform matrices accumulated during data augmentation.

    Each matrix ``M`` is a ``(3, 3)`` numpy array such that applying ``M`` to
    augmented coordinates recovers the original coordinates.  Only the keys
    corresponding to augmentation operations that were actually applied are
    present.

    Fields
    ------
    flip : np.ndarray, shape (3, 3), dtype float32
        Inverse of the flip transformation (diagonal matrix with ±1 entries).
    rotate : np.ndarray, shape (3, 3), dtype float32
        Inverse of the global yaw-rotation matrix (transpose of the rotation).
    rescale : np.ndarray, shape (3, 3), dtype float32
        Inverse of the uniform scale transform (diagonal with ``1/s`` entries).
    translate : np.ndarray, shape (3, 3), dtype float32
        Inverse of the global translation (negated offset stored as matrix).
    """

    flip: np.ndarray
    """shape: (3, 3) – diagonal matrix with ±1 entries; maps augmented coords back to
    original; ego-vehicle frame, dtype float32"""
    rotate: np.ndarray
    """shape: (3, 3) – inverse yaw-rotation (i.e. transpose of the applied rotation
    matrix); ego-vehicle frame, dtype float32"""
    rescale: np.ndarray
    """shape: (3, 3) – diagonal matrix with entries 1/s; undoes the uniform scale
    augmentation; ego-vehicle frame, dtype float32"""
    translate: np.ndarray
    """shape: (3, 3) – encodes negated translation offset; undoes the global
    translation augmentation; ego-vehicle frame, dtype float32"""


class MultiScale3DFeatures(TypedDict, total=False):
    """Sparse 3-D feature maps at four resolution levels produced by Backbone3D.

    Keys correspond to the stride-1 (``x_conv1``) through stride-8
    (``x_conv4``) outputs of ``VoxelBackBone8x`` / ``VoxelResBackBone8x``.
    Each value is a ``spconv.SparseConvTensor``.

    Fields
    ------
    x_conv1 : SparseConvTensor, spatial stride 1
        Highest-resolution sparse feature map.
    x_conv2 : SparseConvTensor, spatial stride 2
    x_conv3 : SparseConvTensor, spatial stride 4
    x_conv4 : SparseConvTensor, spatial stride 8
        Lowest-resolution sparse feature map.
    """

    x_conv1: Any
    """spconv.SparseConvTensor (stride-1 highest-resolution sparse feature map)."""
    x_conv2: Any
    """spconv.SparseConvTensor (stride-2 sparse feature map)."""
    x_conv3: Any
    """spconv.SparseConvTensor (stride-4 sparse feature map)."""
    x_conv4: Any
    """spconv.SparseConvTensor (stride-8 lowest-resolution sparse feature map)."""


class MultiScale3DStrides(TypedDict, total=False):
    """Integer spatial-stride values corresponding to :class:`MultiScale3DFeatures`.

    Fields
    ------
    x_conv1 : int
        Stride 1 (full resolution).
    x_conv2 : int
        Stride 2.
    x_conv3 : int
        Stride 4.
    x_conv4 : int
        Stride 8.
    """

    x_conv1: int
    x_conv2: int
    x_conv3: int
    x_conv4: int


class PointFeaturesDict(TypedDict, total=False):
    """Per-point feature tensors keyed by feature-location name.

    Used by :class:`PDVHead` and :class:`VoxelCenterHead` to pass per-voxel /
    per-point features to the ROI grid-pooling layers.

    Fields
    ------
    x_conv1 : torch.Tensor, shape (N1, C1), dtype float32
        Point features from the stride-1 3-D backbone layer.
    x_conv2 : torch.Tensor, shape (N2, C2), dtype float32
        Point features from the stride-2 layer.
    x_conv3 : torch.Tensor, shape (N3, C3), dtype float32
        Point features from the stride-4 layer.
    x_conv4 : torch.Tensor, shape (N4, C4), dtype float32
        Point features from the stride-8 layer.
    """

    x_conv1: torch.Tensor
    """shape: (N1, C1) – N1 active voxels at stride-1; C1 feature channels;
    LiDAR/ego-vehicle frame, dtype float32"""
    x_conv2: torch.Tensor
    """shape: (N2, C2) – N2 active voxels at stride-2; C2 feature channels;
    LiDAR/ego-vehicle frame, dtype float32"""
    x_conv3: torch.Tensor
    """shape: (N3, C3) – N3 active voxels at stride-4; C3 feature channels;
    LiDAR/ego-vehicle frame, dtype float32"""
    x_conv4: torch.Tensor
    """shape: (N4, C4) – N4 active voxels at stride-8; C4 feature channels;
    LiDAR/ego-vehicle frame, dtype float32"""


class PointCoordsDict(TypedDict, total=False):
    """Per-point coordinate tensors keyed by feature-location name.

    Companion to :class:`PointFeaturesDict`.  Each entry contains the 3-D
    world coordinates (and batch index) of the points whose features appear at
    the same key in :class:`PointFeaturesDict`.

    Fields
    ------
    x_conv1 : torch.Tensor, shape (N1, 4), dtype float32
        ``[batch_idx, x, y, z]`` for each point at stride-1 resolution.
    x_conv2 : torch.Tensor, shape (N2, 4), dtype float32
        Same layout at stride-2 resolution.
    x_conv3 : torch.Tensor, shape (N3, 4), dtype float32
        Same layout at stride-4 resolution.
    x_conv4 : torch.Tensor, shape (N4, 4), dtype float32
        Same layout at stride-8 resolution.
    """

    x_conv1: torch.Tensor
    """shape: (N1, 4) – [batch_idx, x, y, z] for each point at stride-1; x/y/z in
    metres, ego-vehicle/LiDAR frame, dtype float32"""
    x_conv2: torch.Tensor
    """shape: (N2, 4) – [batch_idx, x, y, z] for each point at stride-2; x/y/z in
    metres, ego-vehicle/LiDAR frame, dtype float32"""
    x_conv3: torch.Tensor
    """shape: (N3, 4) – [batch_idx, x, y, z] for each point at stride-4; x/y/z in
    metres, ego-vehicle/LiDAR frame, dtype float32"""
    x_conv4: torch.Tensor
    """shape: (N4, 4) – [batch_idx, x, y, z] for each point at stride-8; x/y/z in
    metres, ego-vehicle/LiDAR frame, dtype float32"""


# ---------------------------------------------------------------------------
# Single-sample data dictionary (dataset / augmentor / processor pipeline)
# ---------------------------------------------------------------------------

class DataDictBase(TypedDict):
    """Required fields that are always present in the per-sample data dict."""

    points: np.ndarray
    """Point cloud for one LiDAR sweep (or merged sweeps).

    Shape: ``(N, 3 + C_in)`` where the first three columns are ``[x, y, z]``
    in the ego-vehicle coordinate frame, followed by optional channels such as
    intensity and time offset.  dtype: float32.
    """


class DataDict(DataDictBase, total=False):
    """Per-sample dictionary flowing through the dataset / augmentation pipeline.

    Created by :meth:`DatasetTemplate.__getitem__` and passed through
    :class:`DataAugmentor`, :class:`PointFeatureEncoder`, and
    :class:`DataProcessor`.  After collation it becomes :class:`BatchDict`.

    Required fields
    ---------------
    points : np.ndarray, shape (N, 3+C_in)
        See :class:`DataDictBase`.

    Optional fields (present only during training or after specific steps)
    ----------------------------------------------------------------------
    frame_id : int or str
        Unique identifier for the LiDAR frame within its sequence.
    pose : np.ndarray, shape (4, 4), dtype float64
        Ego-to-world SE(3) transform for this frame.
    sequence_name : str
        Identifier of the driving sequence this frame belongs to.
    gt_names : np.ndarray, shape (M,), dtype '<U32'
        Class name string for each ground-truth object.  Present during
        training and removed by :meth:`DatasetTemplate.prepare_data` after
        class filtering.
    gt_boxes : np.ndarray, shape (M, 7+C), dtype float32
        Ground-truth 3-D boxes ``[x, y, z, dx, dy, dz, heading, ...]``.
        After augmentation the last column is replaced by the 1-indexed class
        label (int).
    gt_boxes_mask : np.ndarray, shape (M,), dtype bool
        ``True`` for GT boxes whose class is in the active class list; used
        inside :class:`DataAugmentor` to filter irrelevant boxes.
    use_lead_xyz : bool
        ``True`` when the xyz columns of ``points`` should be kept as
        point-wise features (set by :class:`PointFeatureEncoder`).
    voxels : np.ndarray, shape (num_voxels, max_pts_per_voxel, C), dtype float32
        Voxelised point-cloud; produced by :class:`DataProcessor`.
    voxel_coords : np.ndarray, shape (num_voxels, 3), dtype int32
        Integer voxel indices ``[z_idx, y_idx, x_idx]`` for each voxel.
    voxel_num_points : np.ndarray, shape (num_voxels,), dtype int32
        Number of points in each voxel (before padding).
    aug_matrix_inv : AugMatrixInv
        Inverse-transform matrices for each augmentation that was applied; used
        during evaluation to map predictions back to the original frame.  Only
        present when at least one augmentation requested noise-return.
    calib : Any
        Sensor-calibration object (dataset-specific).  Populated by some
        dataset loaders and removed by :meth:`DataAugmentor.forward` before
        the sample reaches the model.
    road_plane : Any
        Ground-plane parameters used by certain augmentation operations.
        Populated by some dataset loaders and removed by
        :meth:`DataAugmentor.forward` before the sample reaches the model.
    tta_original : Any
        The unaugmented version of this sample, stored alongside the TTA
        variants when test-time augmentation is active.  Keyed ``"tta_original"``
        inside the per-sample dict produced by :class:`TestTimeAugmentor`.
        Typed as ``Any`` to avoid a self-referential TypedDict definition.
    """

    frame_id: FrameId
    pose: np.ndarray
    """shape: (4, 4) – ego-to-world SE(3) transform for this frame; dtype float64"""
    sequence_name: str
    gt_names: np.ndarray
    """shape: (M,) – class-name string per GT object (e.g. 'Vehicle'); dtype '<U32'"""
    gt_boxes: np.ndarray
    """shape: (M, 7+C) or (M, 8) after prepare_data – [x, y, z, dx, dy, dz, heading,
    (class_label)] in metres/radians; ego-vehicle/LiDAR frame; dtype float32"""
    gt_boxes_mask: np.ndarray
    """shape: (M,) – True for GT boxes whose class is in the active class list; dtype
    bool"""
    use_lead_xyz: bool
    voxels: np.ndarray
    """shape: (num_voxels, max_pts_per_voxel, C) – voxelised point-cloud; C features
    per point (xyz + optional channels); dtype float32"""
    voxel_coords: np.ndarray
    """shape: (num_voxels, 3) – integer voxel grid indices [z_idx, y_idx, x_idx] per
    voxel; dtype int32"""
    voxel_num_points: np.ndarray
    """shape: (num_voxels,) – number of real points in each voxel (before padding);
    dtype int32"""
    aug_matrix_inv: AugMatrixInv
    calib: Any
    road_plane: Any
    tta_original: Any


# ---------------------------------------------------------------------------
# Batched pipeline dictionary (model forward pass)
# ---------------------------------------------------------------------------

class BatchDictBase(TypedDict):
    """Fields that are always present once a batch has been collated."""

    batch_size: int
    """Number of samples in the current mini-batch (scalar int)."""


class BatchDict(BatchDictBase, total=False):
    """Batched dictionary flowing through the full model forward pass.

    Built by :meth:`DatasetTemplate.collate_batch` and then mutated in-place
    by each module in the pipeline:
    VFE → Backbone3D → HeightCompression → Backbone2D → DenseHead → (RoIHead).

    Required fields
    ---------------
    batch_size : int
        See :class:`BatchDictBase`.

    Raw voxelisation inputs (from collation)
    ----------------------------------------
    points : np.ndarray or torch.Tensor, shape (sum_N, 4+C), dtype float32
        Concatenated point clouds with a batch-index prepended:
        ``[batch_idx, x, y, z, ...]``.
    voxels : np.ndarray, shape (total_voxels, max_pts, C), dtype float32
        Voxelised point-cloud data for the whole batch.
    voxel_coords : np.ndarray, shape (total_voxels, 4), dtype int32
        ``[batch_idx, z_idx, y_idx, x_idx]`` for each voxel.
    voxel_num_points : np.ndarray, shape (total_voxels,), dtype int32
        Number of real points in each voxel.
    use_lead_xyz : bool
        Whether xyz was retained as point-wise input features.

    Metadata (from collation)
    -------------------------
    frame_id : np.ndarray of str/int, shape (B,)
        Per-sample frame identifiers.
    pose : np.ndarray, shape (B, 4, 4), dtype float64
        Ego-to-world SE(3) transform for each sample.
    sequence_name : np.ndarray of str, shape (B,)
        Per-sample sequence identifiers.

    Supervision targets (training only)
    ------------------------------------
    gt_boxes : np.ndarray or torch.Tensor, shape (B, M_max, 7+C+1), dtype float32
        Ground-truth boxes zero-padded to the maximum object count.
        Last column is the 1-indexed class label.

    VFE output
    ----------
    voxel_features : torch.Tensor, shape (total_voxels, C_vfe), dtype float32
        Per-voxel feature vectors produced by the VFE (e.g. mean pooling).

    Backbone3D outputs
    ------------------
    encoded_spconv_tensor : SparseConvTensor
        Final sparse 3-D feature tensor output by Backbone3D (stride 8).
    encoded_spconv_tensor_stride : int
        Spatial downsampling factor of ``encoded_spconv_tensor`` (typically 8).
    multi_scale_3d_features : MultiScale3DFeatures
        Intermediate sparse feature maps at four resolution levels.
    multi_scale_3d_strides : MultiScale3DStrides
        Corresponding integer stride values.

    HeightCompression output
    ------------------------
    spatial_features : torch.Tensor, shape (B, C*D, H, W), dtype float32
        Dense BEV feature map produced by collapsing the 3-D sparse tensor
        along the height (D) axis.
    spatial_features_stride : int
        Spatial stride of ``spatial_features`` relative to the voxel grid.

    Backbone2D output
    -----------------
    spatial_features_2d : torch.Tensor, shape (B, C_bev, H', W'), dtype float32
        Final BEV feature map after the 2-D backbone and FPN upsample blocks.

    DenseHead outputs (first-stage or single-stage)
    ------------------------------------------------
    final_box_dicts : list of PredictionDict, length B
        Post-NMS predictions produced by CenterHead (single-stage path).
    rois : torch.Tensor, shape (B, num_rois, 7+C), dtype float32
        Top-K region proposals passed to the ROI head.  Coordinates are in the
        LiDAR frame ``[x, y, z, dx, dy, dz, heading, ...]``.
    roi_scores : torch.Tensor, shape (B, num_rois), dtype float32
        Confidence scores for the proposals in ``rois``.
    roi_labels : torch.Tensor, shape (B, num_rois), dtype int64
        1-indexed class labels for the proposals in ``rois``.
    roi_features : torch.Tensor, shape (B, num_rois, C_feat), dtype float32
        Bilinearly-interpolated BEV features at proposal centres (first stage).
    has_class_labels : bool
        ``True`` when ``roi_labels`` contains valid class predictions.

    PDV / ROI head outputs (second-stage)
    --------------------------------------
    batch_cls_preds : torch.Tensor, shape (B, num_rois, num_classes) or
                      (sum_rois, num_classes), dtype float32
        Per-ROI class logits produced by the ROI head.
    batch_box_preds : torch.Tensor, shape (B, num_rois, 7+C) or
                      (sum_rois, 7+C), dtype float32
        Per-ROI box predictions produced by the ROI head.
    cls_preds_normalized : bool
        ``True`` when ``batch_cls_preds`` has already been passed through a
        sigmoid / softmax.
    batch_index : torch.Tensor, shape (sum_rois,), dtype int64
        Batch index for each prediction when predictions are stored in a flat
        (not B × N) layout.

    Point features (PDV head / VoxelCenterHead only)
    -------------------------------------------------
    point_features : PointFeaturesDict
        Per-voxel/point feature tensors keyed by backbone layer name.
    point_coords : PointCoordsDict
        Corresponding ``[batch_idx, x, y, z]`` coordinates.

    TTA fields
    ----------
    tta_ops : list of str
        Names of the test-time augmentation variants present in the batch,
        e.g. ``['tta_original', 'tta_flip_x', ...]``.
    multihead_label_mapping : list of torch.Tensor
        Per-head label index mapping used during multi-head NMS.
    """

    # raw inputs
    points: Union[np.ndarray, torch.Tensor]
    """shape: (sum_N, 1+3+C_in) – batch-concatenated points; columns [batch_idx, x, y,
    z, ...]; x/y/z in metres; ego-vehicle/LiDAR frame; dtype float32"""
    voxels: np.ndarray
    """shape: (total_voxels, max_pts, C) – voxelised point-cloud for the whole batch; C
    features per point; dtype float32"""
    voxel_coords: np.ndarray
    """shape: (total_voxels, 4) – [batch_idx, z_idx, y_idx, x_idx] per voxel across the
    batch; dtype int32"""
    voxel_num_points: np.ndarray
    """shape: (total_voxels,) – number of real points in each voxel (before padding);
    dtype int32"""
    use_lead_xyz: bool

    # metadata
    frame_id: np.ndarray
    """shape: (B,) – per-sample frame identifiers; dtype str or int"""
    pose: np.ndarray
    """shape: (B, 4, 4) – ego-to-world SE(3) transform for each sample; dtype float64"""
    sequence_name: np.ndarray
    """shape: (B,) – per-sample sequence identifiers; dtype str"""

    # supervision
    gt_boxes: Union[np.ndarray, torch.Tensor]
    """shape: (B, M_max, 7+C+1) – GT boxes zero-padded to max object count; [x, y, z,
    dx, dy, dz, heading, ..., class_label]; x/y/z in metres, heading in radians;
    ego-vehicle/LiDAR frame; dtype float32"""

    # VFE output
    voxel_features: torch.Tensor
    """shape: (total_voxels, C_vfe) – per-voxel feature vectors from the VFE (e.g. mean
    pooling over points); dtype float32"""

    # Backbone3D outputs
    encoded_spconv_tensor: Any
    """spconv.SparseConvTensor – final sparse 3-D feature volume at stride 8; spatial
    dims in LiDAR/ego-vehicle frame"""
    encoded_spconv_tensor_stride: int
    multi_scale_3d_features: MultiScale3DFeatures
    multi_scale_3d_strides: MultiScale3DStrides

    # HeightCompression output
    spatial_features: torch.Tensor
    """shape: (B, C*D, H, W) – dense BEV feature map; C channels × D height-bins
    collapsed; H/W are spatial BEV dims in LiDAR/ego-vehicle frame; dtype float32"""
    spatial_features_stride: int

    # Backbone2D output
    spatial_features_2d: torch.Tensor
    """shape: (B, C_bev, H', W') – final BEV feature map after 2-D backbone and FPN
    upsample; H'/W' spatial dims in LiDAR/ego-vehicle frame; dtype float32"""

    # DenseHead / first-stage outputs
    final_box_dicts: List["PredictionDict"]
    rois: torch.Tensor
    """shape: (B, num_rois, 7+C) – top-K region proposals; [x, y, z, dx, dy, dz,
    heading, ...]; x/y/z in metres, heading in radians; LiDAR/ego-vehicle frame;
    dtype float32"""
    roi_scores: torch.Tensor
    """shape: (B, num_rois) – confidence scores in [0, 1] for each proposal; dtype
    float32"""
    roi_labels: torch.Tensor
    """shape: (B, num_rois) – 1-indexed class labels for each proposal; dtype int64"""
    roi_features: torch.Tensor
    """shape: (B, num_rois, C_feat) – bilinearly-interpolated BEV features at proposal
    centres; dtype float32"""
    has_class_labels: bool

    # ROI head outputs
    batch_cls_preds: torch.Tensor
    """shape: (B, num_rois, num_classes) or (sum_rois, num_classes) – per-ROI class
    logits (before sigmoid/softmax); dtype float32"""
    batch_box_preds: torch.Tensor
    """shape: (B, num_rois, 7+C) or (sum_rois, 7+C) – per-ROI box predictions; [x, y,
    z, dx, dy, dz, heading, ...]; LiDAR/ego-vehicle frame; dtype float32"""
    cls_preds_normalized: bool
    batch_index: Optional[torch.Tensor]
    """shape: (sum_rois,) – batch index for each prediction in flat (non-padded)
    layout; dtype int64"""

    # Point features (PDV head)
    point_features: PointFeaturesDict
    point_coords: PointCoordsDict

    # TTA
    tta_ops: List[str]
    multihead_label_mapping: List[torch.Tensor]
    """per element shape: (num_classes_i,) – maps head-local class indices to global
    1-indexed labels; dtype int64"""


# ---------------------------------------------------------------------------
# DenseHead (CenterHead) prediction and target dicts
# ---------------------------------------------------------------------------

class SeparateHeadPredDictBase(TypedDict):
    """Required per-head raw outputs always produced by :class:`SeparateHead`."""

    hm: torch.Tensor
    """Heatmap logits (before sigmoid).

    Shape: ``(B, num_classes, H, W)``, dtype float32.
    """

    center: torch.Tensor
    """Sub-voxel centre offsets in the BEV plane.

    Shape: ``(B, 2, H, W)`` where channels are ``[delta_x, delta_y]``,
    dtype float32.
    """

    center_z: torch.Tensor
    """Object centre height (z coordinate in LiDAR frame).

    Shape: ``(B, 1, H, W)``, dtype float32.
    """

    dim: torch.Tensor
    """Log-space box dimensions ``[log(l), log(w), log(h)]``.

    Shape: ``(B, 3, H, W)``, dtype float32.  Apply ``exp()`` to recover
    the metric dimensions.
    """

    rot: torch.Tensor
    """Rotation encoding ``[cos(θ), sin(θ)]``.

    Shape: ``(B, 2, H, W)``, dtype float32.  Use ``atan2`` to recover the
    heading angle θ.
    """


class SeparateHeadPredDict(SeparateHeadPredDictBase, total=False):
    """Per-task raw predictions produced by :class:`SeparateHead`.

    The required fields (hm, center, center_z, dim, rot) are always present.
    The optional fields are included only when the corresponding head is
    configured.

    Optional fields
    ---------------
    vel : torch.Tensor, shape (B, 2, H, W), dtype float32
        Velocity predictions ``[vx, vy]`` in the LiDAR frame.  Present when
        a velocity head is configured (e.g. nuScenes).
    iou : torch.Tensor, shape (B, 1, H, W), dtype float32
        Predicted IoU scores for IoU-aware confidence re-weighting.  Present
        when ``IOU_WEIGHT > 0`` in the model config.
    """

    vel: torch.Tensor
    """shape: (B, 2, H, W) – velocity predictions [vx, vy] in metres/second;
    LiDAR/ego-vehicle frame; dtype float32"""
    iou: torch.Tensor
    """shape: (B, 1, H, W) – predicted IoU score for IoU-aware confidence re-weighting;
    values in [0, 1] after sigmoid; dtype float32"""


class PredictionDict(TypedDict):
    """Post-processed predictions for one sample in the batch.

    Produced by :meth:`CenterHead.generate_predicted_boxes` and
    :meth:`CenterPoint.post_processing`.  Used as elements of the
    ``pred_dicts`` list returned by the model at inference time.

    Fields
    ------
    pred_boxes : torch.Tensor, shape (N, 7+C), dtype float32
        Predicted 3-D bounding boxes ``[x, y, z, dx, dy, dz, heading, ...]``
        in the LiDAR coordinate frame.
    pred_scores : torch.Tensor, shape (N,), dtype float32
        Confidence scores in ``[0, 1]`` for each predicted box.
    pred_labels : torch.Tensor, shape (N,), dtype int64
        1-indexed class labels for each predicted box.
    """

    pred_boxes: torch.Tensor
    """shape: (N, 7+C) – predicted 3-D boxes [x, y, z, dx, dy, dz, heading, ...]; x/y/z
    in metres, dx/dy/dz in metres, heading in radians; LiDAR/ego-vehicle frame;
    dtype float32"""
    pred_scores: torch.Tensor
    """shape: (N,) – confidence scores in [0, 1] for each predicted box; dtype float32"""
    pred_labels: torch.Tensor
    """shape: (N,) – 1-indexed class labels for each predicted box; dtype int64"""


class CenterHeadTargetDict(TypedDict):
    """Heatmap assignment targets produced by :meth:`CenterHead.assign_targets`.

    Each field is a list with one element per detection head (task).

    Fields
    ------
    heatmaps : list of torch.Tensor, each shape (B, num_classes_i, H, W)
        Gaussian heatmaps for the i-th head's object categories.
        dtype float32 in ``[0, 1]``.
    target_boxes : list of torch.Tensor, each shape (B, max_objs, code_size)
        Encoded box regression targets at the Gaussian peaks.
        code_size = 8 (dx, dy, z, log_l, log_w, log_h, cos_θ, sin_θ)
        plus optional velocity channels.  dtype float32.
    inds : list of torch.Tensor, each shape (B, max_objs), dtype int64
        Flattened ``H * W`` index of the voxel assigned to each object.
    masks : list of torch.Tensor, each shape (B, max_objs), dtype int64
        Binary mask; ``1`` for valid objects, ``0`` for padding.
    heatmap_masks : list of torch.Tensor
        Reserved for future use; currently populated as an empty list.
    """

    heatmaps: List[torch.Tensor]
    """list length = num_heads; each shape: (B, num_classes_i, H, W) – Gaussian heatmap
    targets in [0, 1]; H×W is the BEV feature-map grid; LiDAR/ego-vehicle frame;
    dtype float32"""
    target_boxes: List[torch.Tensor]
    """list length = num_heads; each shape: (B, max_objs, code_size) – encoded box
    regression targets at Gaussian peaks; code_size = 8: [delta_x, delta_y, z,
    log(dx), log(dy), log(dz), cos(θ), sin(θ)] + optional vel; dtype float32"""
    inds: List[torch.Tensor]
    """list length = num_heads; each shape: (B, max_objs) – flattened H×W grid index of
    the voxel assigned to each object; dtype int64"""
    masks: List[torch.Tensor]
    """list length = num_heads; each shape: (B, max_objs) – binary validity mask; 1 for
    real objects, 0 for padding; dtype int64"""
    heatmap_masks: List[torch.Tensor]
    """list length = num_heads; reserved for future use; currently populated as empty
    list"""


# ---------------------------------------------------------------------------
# ROI head target and forward-state dicts
# ---------------------------------------------------------------------------

class ProposalTargetDict(TypedDict):
    """ROI sampling targets from :class:`ProposalTargetLayer`.

    Produced by the ``forward`` method of both :class:`ProposalTargetLayer`
    and :class:`ProposalTargetLayer_CP` and consumed by the ROI head's loss
    functions.

    Fields
    ------
    rois : torch.Tensor, shape (B, M, 7+C), dtype float32
        Sampled ROIs (M = ROI_PER_IMAGE).  Coordinates are in the LiDAR
        frame ``[x, y, z, dx, dy, dz, heading, ...]``.
    gt_of_rois : torch.Tensor, shape (B, M, 7+C), dtype float32
        Ground-truth boxes transformed into the canonical coordinate frame
        of each sampled ROI (centred at ROI centre, aligned with ROI heading).
    gt_iou_of_rois : torch.Tensor, shape (B, M), dtype float32
        3-D IoU between each sampled ROI and its best-matching GT box.
    roi_scores : torch.Tensor, shape (B, M), dtype float32
        Confidence scores of the proposal network for each sampled ROI.
    roi_labels : torch.Tensor, shape (B, M), dtype int64
        1-indexed class labels of the sampled ROIs.
    reg_valid_mask : torch.Tensor, shape (B, M), dtype int64
        Binary mask; ``1`` when the ROI's IoU exceeds REG_FG_THRESH and is
        therefore a valid regression target.
    rcnn_cls_labels : torch.Tensor, shape (B, M), dtype float32 or int64
        Classification targets; ``1`` for foreground, ``0`` for background,
        values in (0, 1) for IoU-based soft labels, ``-1`` to ignore.
    """

    rois: torch.Tensor
    """shape: (B, M, 7+C) – sampled ROIs; M = ROI_PER_IMAGE; [x, y, z, dx, dy, dz,
    heading, ...]; x/y/z in metres, heading in radians; LiDAR/ego-vehicle frame;
    dtype float32"""
    gt_of_rois: torch.Tensor
    """shape: (B, M, 7+C) – GT box in canonical frame of each ROI (translated to ROI
    centre, rotated to ROI heading); distances in metres, heading in radians; dtype
    float32"""
    gt_iou_of_rois: torch.Tensor
    """shape: (B, M) – 3-D IoU ∈ [0, 1] between each sampled ROI and its best-matching
    GT box; dtype float32"""
    roi_scores: torch.Tensor
    """shape: (B, M) – confidence scores ∈ [0, 1] from the proposal network for each
    sampled ROI; dtype float32"""
    roi_labels: torch.Tensor
    """shape: (B, M) – 1-indexed class labels for each sampled ROI; dtype int64"""
    reg_valid_mask: torch.Tensor
    """shape: (B, M) – binary mask; 1 when ROI IoU ≥ REG_FG_THRESH (valid regression
    target), 0 otherwise; dtype int64"""
    rcnn_cls_labels: torch.Tensor
    """shape: (B, M) – classification targets; 1 = foreground, 0 = background, (0,1) =
    IoU-based soft label, -1 = ignore; dtype float32 or int64"""


class RoIHeadForwardDictBase(ProposalTargetDict):
    """Fields inherited from :class:`ProposalTargetDict` that are always set."""


class RoIHeadForwardDict(RoIHeadForwardDictBase, total=False):
    """State dictionary stored in ``RoIHeadTemplate.forward_ret_dict``.

    Extends :class:`ProposalTargetDict` with the canonical-frame GT boxes and
    the ROI head's own predictions, all of which are needed by the loss
    functions.

    Additional required fields (always set before loss is computed)
    ---------------------------------------------------------------
    gt_of_rois_src : torch.Tensor, shape (B, M, 7+C), dtype float32
        Original (pre-canonical-transform) GT boxes corresponding to each
        sampled ROI; used for corner-loss regularisation.
    rcnn_cls : torch.Tensor, shape (B*M, num_class) or (B*M, 1), dtype float32
        ROI head classification logits (before sigmoid/softmax).
    rcnn_reg : torch.Tensor, shape (B*M, code_size), dtype float32
        ROI head box regression predictions in the canonical frame.

    Optional fields (present only with ProposalTargetLayer_CP)
    ----------------------------------------------------------
    roi_features : torch.Tensor, shape (B, M, C_feat), dtype float32
        Per-ROI feature vectors carried through ROI sampling (CP variant).
    """

    gt_of_rois_src: torch.Tensor
    """shape: (B, M, 7+C) – original (pre-canonical-transform) GT boxes for each
    sampled ROI; used for corner-loss; LiDAR/ego-vehicle frame; dtype float32"""
    rcnn_cls: torch.Tensor
    """shape: (B*M, num_class) or (B*M, 1) – ROI head classification logits before
    sigmoid/softmax; num_class or 1 for binary; dtype float32"""
    rcnn_reg: torch.Tensor
    """shape: (B*M, code_size) – ROI head box regression predictions in canonical ROI
    frame; code_size = 7 ([dx, dy, dz, log_l, log_w, log_h, dθ]); dtype float32"""
    roi_features: torch.Tensor
    """shape: (B, M, C_feat) – per-ROI feature vectors carried through ROI sampling (CP
    variant only); C_feat = feature dimension; dtype float32"""


# ---------------------------------------------------------------------------
# Model-building info dict
# ---------------------------------------------------------------------------

class ModelInfoDictBase(TypedDict):
    """Minimum set of fields required to build the first module."""

    module_list: List[Any]
    """Accumulator for constructed nn.Module instances (built in order)."""

    num_point_features: int
    """Number of point-wise feature channels flowing into/out of the current
    module under construction; updated after each module is built."""

    grid_size: np.ndarray
    """Voxel-grid dimensions ``[Nx, Ny, Nz]`` (int64), derived from the
    point-cloud range and voxel size."""

    point_cloud_range: np.ndarray
    """Axis-aligned bounding box of the detection volume:
    ``[x_min, y_min, z_min, x_max, y_max, z_max]``, dtype float32."""

    voxel_size: np.ndarray
    """Physical size of one voxel ``[vx, vy, vz]`` in metres, dtype float32."""


class ModelInfoDict(ModelInfoDictBase, total=False):
    """Architecture-info dictionary used during :meth:`CenterPoint.build_networks`.

    Passed by reference and updated after each sub-module is constructed so
    that subsequent modules can query the output dimensionality of preceding
    ones.

    Optional fields (added progressively during network construction)
    -----------------------------------------------------------------
    num_bev_features : int
        Number of BEV feature channels; set after the MAP_TO_BEV and
        BACKBONE_2D modules are constructed.
    """

    num_bev_features: int


# ---------------------------------------------------------------------------
# Evaluation / annotation output dicts
# ---------------------------------------------------------------------------

# AnnotationDict is imported from detzero_utils.structures (see top of file).
# It is the per-frame detection output record and forms the shared interface
# between the detection and tracking modules.


class RecallDict(TypedDict, total=False):
    """Recall-metric accumulator used in :meth:`CenterPoint.post_processing`.

    Keys are generated dynamically from the configured IoU threshold list.
    All values are cumulative counts (int) accumulated across the batch.

    Fields
    ------
    gt : int
        Total number of ground-truth objects seen so far.
    roi_<thresh> : int
        Number of GT objects recalled by first-stage ROI proposals at the
        given IoU threshold (e.g. ``roi_0.3``, ``roi_0.5``, ``roi_0.7``).
    rcnn_<thresh> : int
        Number of GT objects recalled by the final (RCNN) predictions at the
        given IoU threshold.

    Note: the dynamic ``roi_*`` / ``rcnn_*`` keys cannot be expressed as
    static TypedDict fields; they are documented here for reference.  In
    practice the dict is accessed as a regular :class:`dict` for the dynamic
    keys.
    """

    gt: int


# ---------------------------------------------------------------------------
# Public API of this module
# ---------------------------------------------------------------------------

__all__ = [
    # Base classes (required-fields-only layers)
    "DataDictBase",
    "BatchDictBase",
    "SeparateHeadPredDictBase",
    "RoIHeadForwardDictBase",
    "ModelInfoDictBase",
    # Full public types
    "AugMatrixInv",
    "MultiScale3DFeatures",
    "MultiScale3DStrides",
    "PointFeaturesDict",
    "PointCoordsDict",
    "DataDict",
    "BatchDict",
    "SeparateHeadPredDict",
    "PredictionDict",
    "CenterHeadTargetDict",
    "ProposalTargetDict",
    "RoIHeadForwardDict",
    "ModelInfoDict",
    "AnnotationDict",
    "RecallDict",
]
