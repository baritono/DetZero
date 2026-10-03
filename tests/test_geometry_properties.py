"""Property-based tests for geometry helpers.

Each test states an invariant that must hold for *all* inputs; Hypothesis
searches for counter-examples and shrinks them.  Shape contracts are enforced
on every call (see conftest), so these also exercise the annotations.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
import hypothesis.extra.numpy as hnp

from detzero_utils import box_utils, common_utils



def f32(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Finite float32-representable values in [lo, hi]."""
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False, width=32)
coord = f32(-100, 100)
size = f32(0.5, 20)
angle = f32(-10, 10)


@st.composite
def boxes7(draw, min_n=0, max_n=16):
    n = draw(st.integers(min_n, max_n))
    centers = draw(hnp.arrays(np.float32, (n, 3), elements=coord))
    dims = draw(hnp.arrays(np.float32, (n, 3), elements=size))
    heading = draw(hnp.arrays(np.float32, (n, 1), elements=angle))
    return np.concatenate([centers, dims, heading], axis=1)


# --------------------------------------------------------------------------- #
# common_utils
# --------------------------------------------------------------------------- #

@given(
    pts=hnp.arrays(np.float32, hnp.array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=8)
                   .filter(lambda s: s[2] >= 3), elements=coord),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_rotate_points_preserves_z_norm_and_extra_channels(pts, data):
    a = data.draw(hnp.arrays(np.float32, (pts.shape[0],), elements=angle))
    out = common_utils.rotate_points_along_z(pts, a)
    assert out.shape == pts.shape
    np.testing.assert_allclose(out[..., 2:], pts[..., 2:], rtol=0, atol=0)
    np.testing.assert_allclose(
        np.linalg.norm(out[..., :2], axis=-1), np.linalg.norm(pts[..., :2], axis=-1),
        rtol=1e-4, atol=1e-3,
    )


@given(
    pts=hnp.arrays(np.float32, st.tuples(st.integers(1, 4), st.integers(1, 8), st.just(3)),
                   elements=coord),
    data=st.data(),
)
@settings(max_examples=100, deadline=None)
def test_rotate_points_roundtrip(pts, data):
    a = data.draw(hnp.arrays(np.float32, (pts.shape[0],), elements=angle))
    back = common_utils.rotate_points_along_z(common_utils.rotate_points_along_z(pts, a), -a)
    np.testing.assert_allclose(back, pts, rtol=1e-4, atol=1e-3)


@given(
    val=hnp.arrays(np.float32, hnp.array_shapes(max_dims=2, max_side=8),
                   elements=f32(-1e3, 1e3)),
    offset=st.sampled_from([0.0, 0.5, 1.0]),
)
@settings(max_examples=100, deadline=None)
def test_limit_period_lands_in_range(val, offset):
    period = 2 * np.pi
    out = common_utils.limit_period(val, offset=offset, period=period)
    assert out.shape == val.shape
    lo, hi = -offset * period, (1 - offset) * period
    tol = 1e-3
    assert np.all(out >= lo - tol) and np.all(out <= hi + tol)
    # equal to the input modulo the period
    k = (val - out) / period
    np.testing.assert_allclose(k, np.round(k), atol=1e-3)


@given(pts=hnp.arrays(np.float64, st.tuples(st.integers(0, 16), st.just(3)), elements=coord))
@settings(max_examples=100, deadline=None)
def test_cylinder_roundtrip(pts):
    back = common_utils.cylinder2cart(common_utils.cart2cylinder(pts))
    np.testing.assert_allclose(back, pts, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# box_utils
# --------------------------------------------------------------------------- #

@given(boxes=boxes7())
@settings(max_examples=100, deadline=None)
def test_corners_are_centred_on_box(boxes):
    corners = box_utils.boxes_to_corners_3d(boxes)
    assert corners.shape == (boxes.shape[0], 8, 3)
    np.testing.assert_allclose(corners.mean(axis=1), boxes[:, :3], rtol=1e-4, atol=1e-3)


@given(boxes=boxes7())
@settings(max_examples=100, deadline=None)
def test_corner_edge_lengths_match_dims(boxes):
    c = box_utils.boxes_to_corners_3d(boxes)
    # corner order: 0-1 spans dy, 0-3 spans dx, 0-4 spans dz (see docstring diagram)
    np.testing.assert_allclose(np.linalg.norm(c[:, 0] - c[:, 3], axis=-1), boxes[:, 3], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(c[:, 0] - c[:, 1], axis=-1), boxes[:, 4], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(c[:, 0] - c[:, 4], axis=-1), boxes[:, 5], rtol=1e-4, atol=1e-3)


@given(a=boxes7(min_n=1), b=boxes7(min_n=1))
@settings(max_examples=100, deadline=None)
def test_nearest_bev_iou_bounds_symmetry_and_self(a, b):
    ta, tb = torch.from_numpy(a), torch.from_numpy(b)
    iou_ab = box_utils.boxes3d_nearest_bev_iou(ta, tb)
    iou_ba = box_utils.boxes3d_nearest_bev_iou(tb, ta)
    assert iou_ab.shape == (len(a), len(b))
    assert torch.all(iou_ab >= 0) and torch.all(iou_ab <= 1 + 1e-5)
    torch.testing.assert_close(iou_ab, iou_ba.T)
    self_iou = box_utils.boxes3d_nearest_bev_iou(ta, ta).diagonal()
    torch.testing.assert_close(self_iou, torch.ones_like(self_iou), rtol=1e-4, atol=1e-4)


@given(boxes=boxes7(), extra=st.tuples(size, size, size))
@settings(max_examples=50, deadline=None)
def test_enlarge_box3d_only_touches_dims(boxes, extra):
    out = box_utils.enlarge_box3d(boxes, extra_width=extra)
    assert isinstance(out, torch.Tensor) and out.shape == boxes.shape
    np.testing.assert_allclose(out[:, :3].numpy(), boxes[:, :3])
    np.testing.assert_allclose(out[:, 6].numpy(), boxes[:, 6])
    np.testing.assert_allclose(out[:, 3:6].numpy(), boxes[:, 3:6] + np.array(extra, dtype=np.float32), rtol=1e-6)


# --------------------------------------------------------------------------- #
# Contracts reject wrong shapes at the call site
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [np.zeros((4, 6), np.float32), np.zeros((4,), np.float32),
                                 np.zeros((4, 7), np.int64)])
def test_boxes_to_corners_rejects_bad_input(bad):
    from detzero_utils import shape_types
    if not shape_types.SHAPE_CHECK_ENABLED:
        pytest.skip("shape checking disabled")
    with pytest.raises(Exception, match="boxes3d"):
        box_utils.boxes_to_corners_3d(bad)


def test_rotate_points_rejects_mismatched_batch():
    from detzero_utils import shape_types
    if not shape_types.SHAPE_CHECK_ENABLED:
        pytest.skip("shape checking disabled")
    with pytest.raises(Exception):
        common_utils.rotate_points_along_z(np.zeros((2, 5, 3), np.float32), np.zeros(3, np.float32))
