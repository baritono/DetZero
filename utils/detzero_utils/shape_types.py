"""
Shape contracts as types.

This module is the single place DetZero code imports tensor/array shape
annotations from.  It wraps `jaxtyping <https://github.com/patrick-kidger/jaxtyping>`_
so that a shape mismatch becomes an ordinary type error with a *named*
dimension, instead of a broadcast that silently succeeds 200 lines later.

Usage
-----
.. code-block:: python

    from detzero_utils.shape_types import Float, Tensor, TensorOrArray, shape_checked

    @shape_checked
    def boxes_to_corners_3d(
        boxes3d: Float[TensorOrArray, "N 7"],
    ) -> Float[TensorOrArray, "N 8 3"]:
        ...

Runtime checking
----------------
Annotations are always present (for readers, IDEs, and static checkers), but
they are only *enforced* when the environment variable ``DETZERO_SHAPE_CHECK``
is set to ``1`` before the annotated module is imported and both ``jaxtyping``
and ``beartype`` are installed.  The test suite turns this on (see
``tests/conftest.py``); training / inference runs keep it off so there is zero
overhead on the hot path.

If ``jaxtyping`` is not installed (it is a dev-only dependency), the dtype
markers below degrade to ``typing.Annotated`` aliases that carry the shape
string as metadata, so importing annotated modules never fails.

Dimension-name conventions
--------------------------
Use these names consistently so contracts compose across function boundaries
(the same name inside one call must bind to the same size):

=================  ==========================================================
``B``              batch size
``N`` / ``M``      number of boxes or points (``M`` for a second, independent set)
``K``              number of selected / top-k items
``P``              number of points (when ``N`` is already used for boxes)
``T``              number of time-steps in a track
``box_dim``        box code size, ``7 + C`` ([x, y, z, dx, dy, dz, heading, ...])
``point_dim``      point feature size, ``3 + C`` ([x, y, z, ...])
``code_size``      regression code size of a box coder
``num_class``      number of classes
``H`` / ``W``      feature-map height / width
``*batch``         any number of leading dimensions (jaxtyping variadic)
=================  ==========================================================

Write a single-dimension shape with a leading space, e.g. ``Float[Tensor, " N"]``:
jaxtyping strips it, and it stops linters from reading ``"N"`` as a forward
reference to an undefined name (ruff/pyflakes F821).
"""

import functools
import os
from typing import Any, Callable, Sequence, TypeVar, Union

import numpy as np
from torch import Tensor

try:  # Python >= 3.9
    from typing import Annotated
except ImportError:  # pragma: no cover - Python 3.8
    from typing_extensions import Annotated


__all__ = [
    "Tensor",
    "TensorOrArray",
    "FloatVector",
    "Float",
    "Int",
    "Bool",
    "Shaped",
    "shape_checked",
    "SHAPE_CHECK_ENABLED",
    "HAS_JAXTYPING",
    # canonical aliases
    "Boxes3D",
    "Boxes3DWithVel",
    "BoxesND",
    "BEVBoxes",
    "Corners3D",
    "Points",
    "Pose",
]


TensorOrArray = Union[np.ndarray, Tensor]
"""Inputs accepted by the many DetZero helpers that run on either backend
(via :func:`detzero_utils.common_utils.check_numpy_to_torch`)."""

FloatVector = Union[Sequence[float], np.ndarray, Tensor]
"""Small 1-D config vectors such as ``point_cloud_range`` or ``voxel_size``,
which reach DetZero code as lists, numpy arrays or tensors."""


try:
    from jaxtyping import Bool, Float, Int, Shaped, jaxtyped

    HAS_JAXTYPING = True
except ImportError:  # pragma: no cover - exercised only without dev deps
    HAS_JAXTYPING = False

    class _DtypeMarker:
        """Fallback for ``Float[...]`` etc. when jaxtyping is unavailable.

        ``Float[Tensor, "N 7"]`` becomes ``Annotated[Tensor, "Float", "N 7"]``:
        a valid type for any static checker, and the shape string is preserved
        as metadata for documentation tools.
        """

        def __init__(self, name: str) -> None:
            self._name = name

        def __getitem__(self, item: Any) -> Any:
            array_type, shape = item
            return Annotated[array_type, self._name, shape]

        def __repr__(self) -> str:
            return self._name

    Float = _DtypeMarker("Float")  # type: ignore[assignment]
    Int = _DtypeMarker("Int")  # type: ignore[assignment]
    Bool = _DtypeMarker("Bool")  # type: ignore[assignment]
    Shaped = _DtypeMarker("Shaped")  # type: ignore[assignment]
    jaxtyped = None  # type: ignore[assignment]

try:
    from beartype import BeartypeConf, beartype

    # ``is_pep484_tower``: an ``int`` satisfies a ``float`` hint, as in mypy.
    _beartype = beartype(conf=BeartypeConf(is_pep484_tower=True))
    HAS_BEARTYPE = True
except ImportError:  # pragma: no cover - exercised only without dev deps
    _beartype = None  # type: ignore[assignment]
    HAS_BEARTYPE = False


SHAPE_CHECK_ENABLED = (
    os.environ.get("DETZERO_SHAPE_CHECK", "0").lower() in ("1", "true", "yes")
    and HAS_JAXTYPING
    and HAS_BEARTYPE
)
"""Whether :func:`shape_checked` enforces contracts at call time.

Read once at import; set ``DETZERO_SHAPE_CHECK=1`` *before* importing DetZero
modules to enable it."""


_F = TypeVar("_F", bound=Callable[..., Any])


def shape_checked(fn: _F) -> _F:
    """Enforce the jaxtyping shape/dtype annotations of ``fn`` at call time.

    A no-op (returns ``fn`` unchanged) unless :data:`SHAPE_CHECK_ENABLED`.
    When enabled, every call checks that annotated arguments and the return
    value match their declared dtype kind and shape, and that every named
    dimension binds to one consistent size across the whole signature.
    """
    if not SHAPE_CHECK_ENABLED:
        return fn
    checked = jaxtyped(typechecker=_beartype)(fn)
    return functools.wraps(fn)(checked)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Canonical aliases.  Prefer these over re-spelling the same contract inline;
# spell it inline when a function needs a second, independent set (``M``).
# --------------------------------------------------------------------------- #

Boxes3D = Float[TensorOrArray, "N 7"]
"""3-D boxes ``[x, y, z, dx, dy, dz, heading]``; (x, y, z) is the box centre."""

Boxes3DWithVel = Float[TensorOrArray, "N 9"]
"""3-D boxes with BEV velocity ``[x, y, z, dx, dy, dz, heading, vx, vy]``."""

BoxesND = Float[TensorOrArray, "N box_dim"]
"""3-D boxes with ``7 + C`` columns; the first seven follow :data:`Boxes3D`."""

BEVBoxes = Float[TensorOrArray, "N 4"]
"""Axis-aligned 2-D boxes ``[x1, y1, x2, y2]``."""

Corners3D = Float[TensorOrArray, "N 8 3"]
"""Eight box corners, ordered as drawn in :func:`box_utils.boxes_to_corners_3d`."""

Points = Float[TensorOrArray, "N point_dim"]
"""Point cloud ``[x, y, z, ...]`` with ``3 + C`` columns."""

Pose = Float[np.ndarray, "4 4"]
"""Homogeneous SE(3) transform (e.g. ego-to-world)."""

