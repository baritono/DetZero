"""
pytest configuration for DetZero smoke tests.

Inserts stub modules for CUDA extensions that cannot be compiled in a CPU-only
CI environment before any test module is imported.  The stubs satisfy import
statements but will raise if any CUDA kernel is actually called, which is fine
because smoke tests only exercise CPU-side Python logic.
"""

import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# PYTHONPATH
# ---------------------------------------------------------------------------
# Allow running `pytest tests/` from the repo root without a full `pip install`.
# Each module lives in its own subdirectory, so we add all four.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("utils", "detection", "tracking", "refining"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# CUDA extension stubs
# ---------------------------------------------------------------------------
# These compiled .so modules don't exist in a CPU-only build.  Register empty
# placeholder modules so that `from detzero_utils.ops.iou3d_nms import
# iou3d_nms_cuda` (and similar) doesn't raise ImportError at collection time.
_CUDA_STUBS = [
    "detzero_utils.ops.iou3d_nms.iou3d_nms_cuda",
    "detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_cuda",
    "detzero_utils.ops.roipoint_pool3d.roipoint_pool3d_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda",
]

for _name in _CUDA_STUBS:
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)


# ---------------------------------------------------------------------------
# spconv stubs (requires CUDA to install; not available in CPU-only CI)
# ---------------------------------------------------------------------------
def _make_spconv_stub(mod_name: str) -> types.ModuleType:
    """Return a stub for spconv sub-modules that exposes nn.Module-compatible classes."""
    import torch.nn as nn

    m = types.ModuleType(mod_name)

    class _Placeholder(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, *args, **kwargs):
            raise RuntimeError(f"{self.__class__.__name__} is a CPU stub; CUDA required")

    for _attr in (
        "SparseModule",
        "SubMConv3d",
        "SparseConv3d",
        "SparseSequential",
        "SparseMaxPool3d",
        "SparseConvTensor",
        "ToDense",
    ):
        setattr(m, _attr, type(_attr, (_Placeholder,), {}))

    # spconv.utils.Point2VoxelCPU3d stub
    class _VoxelGen:
        def __init__(self, *args, **kwargs):
            pass

        def point_to_voxel(self, *args, **kwargs):
            raise RuntimeError("Point2VoxelCPU3d stub; spconv not installed")

    m.Point2VoxelCPU3d = _VoxelGen
    return m


_SPCONV_STUBS = [
    "spconv",
    "spconv.pytorch",
    "spconv.utils",
]

for _name in _SPCONV_STUBS:
    if _name not in sys.modules:
        stub = _make_spconv_stub(_name)
        stub._is_cpu_stub = True  # sentinel so tests can detect the stub
        sys.modules[_name] = stub

# cumm is a spconv dependency imported by detzero_det data_processor
if "cumm" not in sys.modules:
    sys.modules["cumm"] = types.ModuleType("cumm")
if "cumm.tensorview" not in sys.modules:
    sys.modules["cumm.tensorview"] = types.ModuleType("cumm.tensorview")
