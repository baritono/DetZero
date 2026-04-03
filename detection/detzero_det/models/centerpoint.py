import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

from detzero_utils import common_utils
from detzero_utils.ops.iou3d_nms import iou3d_nms_utils

from detzero_det.utils import model_nms_utils
from detzero_det.utils.ensemble_utils.ensemble import wbf_online
from detzero_det.models import centerpoint_modules as cp_modules
from detzero_det.datasets.types import BatchDict
from detzero_det.models.types import ModelInfoDict, PredDict, RecallDict, TrainingOutput


class CenterPoint(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.tta = self.dataset.tta
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.second_stage = model_cfg.SECOND_STAGE
        self.module_list = self.build_networks()

    def forward(
        self, batch_dict: BatchDict
    ) -> Union[TrainingOutput, Tuple[List[PredDict], RecallDict]]:
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
        disp_dict = {}
        tb_dict = {}
        loss, tb_dict = self.dense_head.get_loss()

        if self.second_stage:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn

        return loss, tb_dict, disp_dict

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self) -> List[nn.Module]:
        model_info_dict = {
            'module_list': [],
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
        }

        vfe = cp_modules.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
        )
        model_info_dict['num_point_features'] = vfe.get_output_feature_dim()

        backbone3d = cp_modules.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
        )
        model_info_dict['num_point_features'] = backbone3d.num_point_features

        map_to_bev = cp_modules.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['num_bev_features'] = map_to_bev.num_bev_features

        backbone2d = cp_modules.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['num_bev_features'] = backbone2d.num_bev_features

        dense_head = cp_modules.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            tta=self.tta,
            predict_boxes_when_training=self.second_stage
        )

        self.add_module('vfe', vfe)
        self.add_module('backbone3d', backbone3d)
        self.add_module('map_to_bev', map_to_bev)
        self.add_module('backbone2d', backbone2d)
        self.add_module('dense_head', dense_head)
        module_list = [vfe, backbone3d, map_to_bev, backbone2d, dense_head]

        if self.second_stage:
            roi_head = cp_modules.__all__[self.model_cfg.ROI_HEAD.NAME](
                model_cfg=self.model_cfg.ROI_HEAD,
                input_channels=model_info_dict['num_bev_features'],
                num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
                grid_size=model_info_dict['grid_size'],
                voxel_size=model_info_dict['voxel_size'],
                point_cloud_range=model_info_dict['point_cloud_range']
            )
            self.add_module('roi_head', roi_head)
            module_list.append(roi_head)

        return module_list

    @staticmethod
    def test_time_augment(
        data_dict: BatchDict,
        pred_dicts: List[PredDict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inverse-transform TTA predictions back to the original frame and fuse
        them with Weighted Box Fusion (WBF).

        Args:
            data_dict: BatchDict
                Must contain ``'tta_ops'`` (list of TTA variant names) and
                ``'batch_size'`` (= true batch_size × num_tta_variants).
            pred_dicts: List[PredDict], length = batch_size × num_tta_variants
                Each :class:`PredDict` holds the detections for one
                (sample, TTA-variant) pair.

        Returns:
            boxes:  torch.Tensor, shape (B, N_fused, 7), dtype float32
            scores: torch.Tensor, shape (B, N_fused, 1), dtype float32
            labels: torch.Tensor, shape (B, N_fused, 1), dtype int64
        """
        tta_ops = data_dict["tta_ops"]
        tta_num = len(tta_ops)
        bs = int(data_dict["batch_size"] // tta_num)
        max_num = max([len(p.pred_boxes) for p in pred_dicts])
        box_num = []

        # Pack predictions into padded tensors for vectorised TTA inversion.
        ref: PredDict = pred_dicts[0]
        boxes = torch.zeros(
            (data_dict["batch_size"], max_num, ref.pred_boxes.shape[-1]),
            dtype=ref.pred_boxes.dtype,
            device=ref.pred_boxes.device,
        )
        scores = torch.zeros(
            (data_dict["batch_size"], max_num, 1),
            dtype=ref.pred_scores.dtype,
            device=ref.pred_scores.device,
        )
        labels = torch.zeros(
            (data_dict["batch_size"], max_num, 1),
            dtype=ref.pred_labels.dtype,
            device=ref.pred_labels.device,
        )
        for i, pred in enumerate(pred_dicts):
            n = len(pred.pred_boxes)
            boxes[i, :n, :] = pred.pred_boxes
            scores[i, :n, 0] = pred.pred_scores
            labels[i, :n, 0] = pred.pred_labels
            box_num.append(n)

        # scatter the original and augmented predict boxes
        boxes = boxes.reshape(bs, tta_num, max_num, -1)
        dim = boxes.shape[-1]

        # restore the augmented preds to original coordinates
        for i, tta_cfg in enumerate(tta_ops):
            if tta_cfg == "tta_original":
                continue
            name, param = tta_cfg.split("_")[1], tta_cfg.split("_")[2]

            if name == "flip":
                if param == "x":
                    boxes[:, i, :, 1] = -boxes[:, i, :, 1]
                    boxes[:, i, :, 6] = -boxes[:, i, :, 6]
                    if dim > 7:
                        boxes[:, i, :, 8] = -boxes[:, i, :, 8]
                elif param == "y":
                    boxes[:, i, :, 0] = -boxes[:, i, :, 0]
                    boxes[:, i, :, 6] = -(boxes[:, i, :, 6] + np.pi)
                    if dim > 7:
                        boxes[:, i, :, 7] = -boxes[:, i, :, 7]
                elif param == 'xy':
                    boxes[:, i, :, 0:2] = -boxes[:, i, :, 0:2]
                    boxes[:, i, :, 6] = boxes[:, i, :, 6] + np.pi
                    if dim > 7:
                        boxes[:, i, :, 7:9] = -boxes[:, i, :, 7:9]
            elif name == "rot":
                param = -float(param)
                boxes[:, i, :, 0:3] = common_utils.rotate_points_along_z(
                    boxes[:, i, :, 0:3],
                    np.repeat(np.array([param]), bs)
                )
                boxes[:, i, :, 6] += param
                if dim > 7:
                    boxes[:, i, :, 7:9] = common_utils.rotate_points_along_z(
                        torch.cat([boxes[:, i, :, 7:9],
                                   torch.zeros((bs, max_num, 1),
                                                dtype=dtype,
                                                device=device)], dim=-1),
                        np.repeat(np.array([param]), bs)
                    )[0][:, 0:2]
            elif name == "scale":
                param = float(param)
                boxes[:, i, :, :6] /= param
                if dim > 7:
                    boxes[:, i, :, 7:9] /= param

        # fuse all the results with weighted box fusion
        data_dict["batch_size"] = bs
        boxes = boxes.squeeze(0)
        boxes, scores, labels = wbf_online(boxes, scores, labels)
        return boxes, scores, labels

    def post_processing(self, batch_dict: BatchDict) -> Tuple[List[PredDict], RecallDict]:
        """
        Apply NMS and assemble per-sample :class:`PredDict` results.

        Args:
            batch_dict: BatchDict
                Must contain ``'batch_size'`` and either
                ``'final_box_dicts'`` (first-stage) or
                ``'batch_box_preds'`` / ``'batch_cls_preds'`` (second-stage).

        Returns:
            pred_dicts: List[PredDict], length = batch_size (or 1 when TTA
                is active, as all variants are fused into a single result).
                Each :class:`PredDict` holds:

                * ``pred_boxes``:  shape ``(N, 7)`` float32
                * ``pred_scores``: shape ``(N,)``  float32
                * ``pred_labels``: shape ``(N,)``  int64 (1-based)

            recall_dict: RecallDict — recall accumulators at each IoU
                threshold.
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        
        if self.second_stage:
            recall_dict: RecallDict = {}
            pred_dicts: List[PredDict] = []
            for index in range(batch_size):
                box_preds: torch.Tensor = batch_dict['batch_box_preds'][index]
                src_box_preds = box_preds

                if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                    if not isinstance(cls_preds, list):
                        cls_preds = [cls_preds]
                        multihead_label_mapping = [
                            torch.range(1, self.num_class, device=cls_preds[0].device, dtype=torch.int)]
                    else:
                        multihead_label_mapping = batch_dict['multihead_label_mapping']

                    cur_start_idx = 0
                    pred_scores, pred_labels, pred_boxes = [], [], []
                    for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                        assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                        cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                        cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                            cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                            nms_config=post_process_cfg.NMS_CONFIG,
                            score_thresh=post_process_cfg.SCORE_THRESH
                        )
                        cur_pred_labels = cur_label_mapping[cur_pred_labels]
                        pred_scores.append(cur_pred_scores)
                        pred_labels.append(cur_pred_labels)
                        pred_boxes.append(cur_pred_boxes)
                        cur_start_idx += cur_cls_preds.shape[0]

                    final_scores = torch.cat(pred_scores, dim=0)
                    final_labels = torch.cat(pred_labels, dim=0)
                    final_boxes = torch.cat(pred_boxes, dim=0)
                else:
                    box_preds = batch_dict['batch_box_preds'][index]
                    cls_preds = batch_dict['batch_cls_preds'][index]  # this is the predicted iou
                    label_preds = batch_dict['roi_labels'][index]

                    if batch_dict.get('has_class_labels', False):
                        label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                        label_preds = batch_dict[label_key][index]
                    else:
                        label_preds = label_preds
                    scores = torch.sqrt(torch.sigmoid(cls_preds).reshape(-1) * batch_dict['roi_scores'][index].reshape(-1))
                    mask = (label_preds != 0).reshape(-1)
                    box_preds = box_preds[mask, :]
                    score_preds = scores[mask]
                    label_preds = label_preds[mask]

                    final_scores = score_preds
                    final_labels = label_preds
                    final_boxes = box_preds

                # TODO: check the order of executing TTA
                recall_dict = self.generate_recall_record(
                    box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                    recall_dict=recall_dict,
                    batch_index=0 if self.tta else index,
                    data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )

                pred_dicts.append(PredDict(
                    pred_boxes=final_boxes,
                    pred_scores=final_scores,
                    pred_labels=final_labels,
                ))
        else:
            pred_dicts = batch_dict['final_box_dicts']
            recall_dict = {}
            for index in range(batch_size):
                # TODO: check the order of executing TTA
                recall_dict = self.generate_recall_record(
                    box_preds=pred_dicts[index].pred_boxes,
                    recall_dict=recall_dict,
                    batch_index=0 if self.tta else index,
                    data_dict=batch_dict,
                    thresh_list=post_process_cfg.RECALL_THRESH_LIST
                )

        if not self.training and self.tta:
            final_boxes, final_scores, final_labels = self.test_time_augment(batch_dict, pred_dicts)
            pred_dicts = [PredDict(
                pred_boxes=final_boxes,
                pred_scores=final_scores,
                pred_labels=final_labels,
            )]
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(
        box_preds: torch.Tensor,
        recall_dict: RecallDict,
        batch_index: int,
        data_dict: Optional[BatchDict] = None,
        thresh_list: Optional[List[float]] = None,
    ) -> RecallDict:
        """
        Accumulate recall counts for one sample.

        Args:
            box_preds: torch.Tensor, shape (N, 7), dtype float32
                Predicted boxes (or first-stage ROI proposals) for IoU
                matching.  Only ``[:, :7]`` is used.
            recall_dict: RecallDict
                Running accumulator updated in-place and returned.
            batch_index: int
                Index into ``data_dict['gt_boxes']`` for this sample.
            data_dict: BatchDict or None
                Must contain ``'gt_boxes'`` for recall to be computed.
            thresh_list: List[float] or None
                IoU thresholds at which recall is evaluated (e.g.
                ``[0.3, 0.5, 0.7]``).

        Returns:
            Updated RecallDict with running counts.
        """
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None

        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict
