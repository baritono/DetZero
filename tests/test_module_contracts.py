"""Behaviour of detection / tracking / refining helpers (CPU only)."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp

from detzero_det.utils import box_coder_utils, centernet_utils, loss_utils
from detzero_refine.utils import data_utils as refine_data_utils
from detzero_track.utils import transform_utils



def f32(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Finite float32-representable values in [lo, hi]."""
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False, width=32)


def _random_pose(rng: np.random.Generator) -> np.ndarray:
    yaw = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[:3, 3] = rng.uniform(-50, 50, size=3)
    return pose


# --------------------------------------------------------------------------- #
# detection: box coder
# --------------------------------------------------------------------------- #

@st.composite
def boxes_and_anchors(draw):
    n = draw(st.integers(1, 16))
    c = draw(st.integers(0, 2))
    def arr(lo, hi, cols):
        return draw(hnp.arrays(np.float32, (n, cols), elements=f32(lo, hi)))
    boxes = np.concatenate([arr(-50, 50, 3), arr(0.5, 10, 3), arr(-3, 3, 1), arr(-5, 5, c)], 1)
    anchors = np.concatenate([arr(-50, 50, 3), arr(0.5, 10, 3), arr(-3, 3, 1), arr(-5, 5, c)], 1)
    return torch.from_numpy(boxes), torch.from_numpy(anchors)


@given(ba=boxes_and_anchors(), sincos=st.booleans())
@settings(max_examples=100, deadline=None)
def test_residual_coder_roundtrip(ba, sincos):
    boxes, anchors = ba
    coder = box_coder_utils.ResidualCoder(code_size=boxes.shape[1], encode_angle_by_sincos=sincos)
    enc = coder.encode_torch(boxes.clone(), anchors.clone())
    assert enc.shape == (boxes.shape[0], boxes.shape[1] + int(sincos))
    dec = coder.decode_torch(enc, anchors)
    torch.testing.assert_close(dec[:, :6], boxes[:, :6], rtol=1e-4, atol=1e-3)
    # heading is only recovered modulo 2*pi when encoded as (cos, sin)
    dh = torch.remainder(dec[:, 6] - boxes[:, 6] + np.pi, 2 * np.pi) - np.pi
    assert torch.all(dh.abs() < 1e-3)


def test_residual_coder_batched_leading_dims():
    coder = box_coder_utils.ResidualCoder(code_size=7)
    boxes, anchors = torch.rand(2, 5, 7) + 0.5, torch.rand(2, 5, 7) + 0.5
    assert coder.decode_torch(coder.encode_torch(boxes, anchors), anchors).shape == (2, 5, 7)


# --------------------------------------------------------------------------- #
# detection: losses and CenterNet helpers
# --------------------------------------------------------------------------- #

def test_weighted_smooth_l1_shapes(monkeypatch):
    # __init__ moves code_weights to CUDA unconditionally; keep it on CPU here.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    loss_fn = loss_utils.WeightedSmoothL1Loss(code_weights=[1.0] * 7)
    loss = loss_fn(torch.rand(2, 10, 7), torch.rand(2, 10, 7), torch.rand(2, 10))
    assert loss.shape == (2, 10, 7)


def test_corner_loss_is_zero_for_identical_and_flipped_boxes():
    boxes = torch.rand(6, 7) + 0.5
    flipped = boxes.clone()
    flipped[:, 6] += np.pi
    assert torch.allclose(loss_utils.get_corner_loss_lidar(boxes, boxes), torch.zeros(6), atol=1e-5)
    assert torch.allclose(loss_utils.get_corner_loss_lidar(boxes, flipped), torch.zeros(6), atol=1e-4)


def test_focal_and_reg_loss_shapes():
    pred = torch.rand(2, 3, 8, 8).clamp(1e-4, 1 - 1e-4)
    gt = torch.zeros(2, 3, 8, 8)
    gt[:, :, 4, 4] = 1
    assert loss_utils.neg_loss_cornernet(pred, gt).shape == ()
    reg = loss_utils._reg_loss(torch.rand(2, 5, 4), torch.rand(2, 5, 4), torch.ones(2, 5, dtype=torch.bool))
    assert reg.shape == (4,)


def test_transpose_and_gather_feat_picks_flat_indices():
    feat = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).view(2, 3, 4, 5)
    ind = torch.tensor([[0, 7], [19, 3]])
    out = centernet_utils._transpose_and_gather_feat(feat, ind)
    assert out.shape == (2, 2, 3)
    for b in range(2):
        for k in range(2):
            y, x = divmod(int(ind[b, k]), 5)
            torch.testing.assert_close(out[b, k], feat[b, :, y, x])


def test_bilinear_interpolate_hits_grid_points():
    im = torch.rand(6, 7, 4)
    x = torch.tensor([0.0, 3.0, 6.0])
    y = torch.tensor([0.0, 2.0, 5.0])
    out = centernet_utils.bilinear_interpolate_torch(im, x, y)
    assert out.shape == (3, 4)
    torch.testing.assert_close(out[1], im[2, 3])


def test_gaussian_radius_positive():
    r = centernet_utils.gaussian_radius(torch.rand(5) * 10 + 1, torch.rand(5) * 10 + 1)
    assert r.shape == (5,) and torch.all(r > 0)


def test_decode_bbox_from_heatmap_shapes():
    b, c, h, w, k = 2, 3, 16, 16, 10
    out = centernet_utils.decode_bbox_from_heatmap(
        heatmap=torch.rand(b, c, h, w), rot_cos=torch.rand(b, 1, h, w), rot_sin=torch.rand(b, 1, h, w),
        center=torch.rand(b, 2, h, w), center_z=torch.rand(b, 1, h, w), dim=torch.rand(b, 3, h, w),
        vel=torch.rand(b, 2, h, w), point_cloud_range=[-10, -10, -5, 10, 10, 5], voxel_size=[0.1, 0.1, 0.2],
        feature_map_stride=8, K=k, post_center_limit_range=torch.tensor([-1e3, -1e3, -1e3, 1e3, 1e3, 1e3]),
    )
    assert len(out) == b
    for d in out:
        m = d["pred_boxes"].shape[0]
        assert d["pred_boxes"].shape == (m, 9) and d["pred_scores"].shape == (m,) and m <= k


# --------------------------------------------------------------------------- #
# tracking / refining transforms
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(10))
def test_tracking_transform_roundtrip(seed):
    rng = np.random.default_rng(seed)
    pose = _random_pose(rng)
    boxes = np.concatenate([rng.uniform(-50, 50, (8, 3)), rng.uniform(0.5, 5, (8, 3)),
                            rng.uniform(-np.pi, np.pi, (8, 1))], axis=1)
    there = transform_utils.transform_boxes3d(boxes, pose)
    back = transform_utils.transform_boxes3d(there, pose, inverse=True)
    np.testing.assert_allclose(back[:, :6], boxes[:, :6], atol=1e-4)
    dh = (back[:, 6] - boxes[:, 6] + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(dh, 0, atol=1e-5)


def test_inverse_transform_mat():
    pose = _random_pose(np.random.default_rng(0))
    np.testing.assert_allclose(transform_utils.get_inverse_transform_mat(pose) @ pose, np.eye(4), atol=1e-5)


@given(angle=hnp.arrays(np.float64, st.integers(0, 32), elements=st.floats(-100, 100, allow_nan=False)))
@settings(max_examples=100, deadline=None)
def test_limit_heading_range(angle):
    out = refine_data_utils.limit_heading_range(angle.copy())
    assert out.shape == angle.shape
    assert np.all(out >= -np.pi) and np.all(out < np.pi)


def test_world_to_lidar_inverts_pose():
    rng = np.random.default_rng(1)
    poses = [_random_pose(rng) for _ in range(4)]
    boxes_lidar = np.concatenate([rng.uniform(-50, 50, (4, 3)), rng.uniform(0.5, 5, (4, 3)),
                                  rng.uniform(-1, 1, (4, 1))], axis=1)
    boxes_world = np.stack([
        transform_utils.transform_boxes3d(boxes_lidar[[i]], poses[i])[0] for i in range(4)
    ])
    back = refine_data_utils.world_to_lidar(list(boxes_world), poses)
    np.testing.assert_allclose(back, boxes_lidar, atol=1e-4)
