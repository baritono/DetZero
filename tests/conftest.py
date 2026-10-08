"""Shared pytest setup.

* Puts the four in-repo packages on ``sys.path`` so tests run from a plain
  checkout without ``python setup.py develop``.
* Provides ``<pkg>.version`` when ``setup.py develop`` has not generated it.
* Stubs the compiled CUDA extensions when they are not built, so pure-Python
  code that merely *imports* a module using them stays testable on CPU-only
  machines.  Calling into a stub raises a clear error instead.
"""

import importlib
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("utils", "detection", "tracking", "refining"):
    path = str(REPO_ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

for _pkg in ("detzero_det", "detzero_track", "detzero_refine"):
    _ver = f"{_pkg}.version"
    try:
        importlib.import_module(_ver)
    except ImportError:
        _mod = types.ModuleType(_ver)
        setattr(_mod, "__version__", "0.0.0+test")
        sys.modules[_ver] = _mod

_CUDA_EXTENSIONS = (
    "detzero_utils.ops.iou3d_nms.iou3d_nms_cuda",
    "detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_cuda",
    "detzero_utils.ops.roipoint_pool3d.roipoint_pool3d_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda",
    "detzero_utils.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda",
)


class _MissingExtension(types.ModuleType):
    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        raise RuntimeError(
            f"{self.__name__}.{attr} called, but the compiled extension is not built. "
            "Build it with `python setup.py develop` in the matching package, or mark "
            "the test with @pytest.mark.cuda."
        )


for _name in _CUDA_EXTENSIONS:
    try:
        importlib.import_module(_name)
    except ImportError:
        sys.modules[_name] = _MissingExtension(_name)
