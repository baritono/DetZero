"""The shape-contract machinery itself: aliases, enforcement, error messages."""

import numpy as np
import pytest
import torch

from detzero_utils import shape_types
from detzero_utils.shape_types import Float, Int, Tensor, shape_checked

pytestmark = pytest.mark.skipif(
    not shape_types.SHAPE_CHECK_ENABLED,
    reason="jaxtyping / beartype not installed (pip install -r requirements-dev.txt)",
)

jaxtyping = pytest.importorskip("jaxtyping")
ShapeError = jaxtyping.TypeCheckError


@shape_checked
def _pairwise(
    a: Float[Tensor, "N 7"], b: Float[Tensor, "M 7"],
) -> Float[Tensor, "N M"]:
    return a[:, None, 0] - b[None, :, 0]


@shape_checked
def _same_n(a: Float[Tensor, "N 3"], idx: Int[Tensor, " N"]) -> Float[Tensor, " N"]:
    return a[:, 0] + idx


def test_valid_call_passes():
    out = _pairwise(torch.zeros(4, 7), torch.zeros(5, 7))
    assert out.shape == (4, 5)


def test_wrong_fixed_dim_is_rejected():
    with pytest.raises(ShapeError):
        _pairwise(torch.zeros(4, 6), torch.zeros(5, 7))


def test_named_dim_must_bind_consistently():
    with pytest.raises(ShapeError):
        _same_n(torch.zeros(4, 3), torch.zeros(5, dtype=torch.long))


def test_dtype_kind_is_checked():
    with pytest.raises(ShapeError):
        _pairwise(torch.zeros(4, 7, dtype=torch.long), torch.zeros(5, 7))


def test_wrong_return_shape_is_rejected():
    @shape_checked
    def bad(a: Float[Tensor, "N 7"]) -> Float[Tensor, "N 8 3"]:
        return a

    with pytest.raises(ShapeError):
        bad(torch.zeros(2, 7))


def test_tensor_or_array_accepts_both_backends():
    from detzero_utils.box_utils import boxes_to_corners_3d

    boxes = np.zeros((3, 7), dtype=np.float32)
    assert boxes_to_corners_3d(boxes).shape == (3, 8, 3)
    assert boxes_to_corners_3d(torch.from_numpy(boxes)).shape == (3, 8, 3)


def test_int_satisfies_float_hint():
    from detzero_utils.common_utils import limit_period

    limit_period(np.zeros(3), offset=0, period=2)


def test_shape_checked_preserves_metadata():
    assert _pairwise.__name__ == "_pairwise"
    assert "a" in _pairwise.__annotations__
