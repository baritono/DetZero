"""Regression test: two-stage CenterPoint post-processing keeps one prediction per sample.

A refactor moved ``pred_dicts.append(record_dict)`` out of the per-sample loop in
``CenterPoint.post_processing``, so with batch_size > 1 only the last sample's boxes
survived and ``generate_prediction_dicts`` paired them with sample 0's frame_id.

Runs on CPU without compiled CUDA ops or spconv: both are stubbed when missing,
since ``post_processing`` uses neither.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("utils", "detection"):
    if str(REPO_ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / _sub))


class _Stub(types.ModuleType):
    """Module whose attributes are inert ``nn.Module`` subclasses (usable as base classes)."""

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return type(attr, (torch.nn.Module,), {})


def _ensure_importable(name):
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = _Stub(name)


_ensure_importable("detzero_det.version")
if not hasattr(sys.modules["detzero_det.version"], "__version__"):
    setattr(sys.modules["detzero_det.version"], "__version__", "0.0.0+test")
for _ext in (
    "detzero_utils.ops.iou3d_nms.iou3d_nms_cuda",
    "detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_cuda",
    "detzero_utils.ops.roipoint_pool3d.roipoint_pool3d_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda",
):
    _ensure_importable(_ext)
try:
    import spconv.pytorch  # noqa: F401
except ImportError:
    sys.modules["spconv"] = _Stub("spconv")
    sys.modules["spconv.pytorch"] = _Stub("spconv.pytorch")
    setattr(sys.modules["spconv"], "pytorch", sys.modules["spconv.pytorch"])

from detzero_det.models.centerpoint import CenterPoint  # noqa: E402


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _two_stage_model():
    cfg = _ns(POST_PROCESSING=_ns(NMS_CONFIG=_ns(MULTI_CLASSES_NMS=False), RECALL_THRESH_LIST=[0.5]))
    return _ns(model_cfg=cfg, second_stage=True, training=False, tta=False,
               generate_recall_record=CenterPoint.generate_recall_record)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_second_stage_returns_one_prediction_per_sample(batch_size):
    torch.manual_seed(0)
    num_rois = 5
    batch_dict = {
        "batch_size": batch_size,
        # Sample b's boxes are all filled with b, so each output is traceable to its sample.
        "batch_box_preds": torch.arange(batch_size, dtype=torch.float32).view(-1, 1, 1).expand(-1, num_rois, 9).clone(),
        "batch_cls_preds": torch.randn(batch_size, num_rois, 1),
        "roi_scores": torch.rand(batch_size, num_rois),
        "roi_labels": torch.ones(batch_size, num_rois, dtype=torch.long),
    }

    pred_dicts, _ = CenterPoint.post_processing(_two_stage_model(), batch_dict)

    assert len(pred_dicts) == batch_size
    for b, pred in enumerate(pred_dicts):
        assert pred["pred_boxes"].shape == (num_rois, 9)
        assert torch.all(pred["pred_boxes"] == b), f"sample {b} got another sample's boxes"
        assert pred["pred_scores"].shape == (num_rois,)
        assert pred["pred_labels"].shape == (num_rois,)
