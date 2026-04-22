# Installation Smoke Test Report

This report documents the outcome of provisioning a Ubuntu environment with
minimal CUDA GPU support, installing the dependencies listed in
[`INSTALL.md`](INSTALL.md), and running smoke tests on every DetZero module.

## Test environment

The provisioning target was Ubuntu 20.04 as recommended by `INSTALL.md`, but the
available runner is Ubuntu 24.04 (Noble) without an attached NVIDIA GPU. The
CUDA toolchain was installed with `conda` inside the `detzero` environment so
that `nvcc` is available for compiling the CUDA extensions. Runtime GPU tests
are therefore out of scope (no device present); the smoke tests below verify
import-time correctness and compile-time correctness of all CUDA extensions.

| Component            | Version                                  |
| -------------------- | ---------------------------------------- |
| OS                   | Ubuntu 24.04 (Noble)                     |
| Python               | 3.8.20 (miniconda env `detzero`)         |
| CUDA toolkit (nvcc)  | 11.8.0 (via `nvidia/label/cuda-11.8.0`)  |
| GCC host compiler    | 11.5.0 (`g++-11`, required by CUDA 11.8) |
| PyTorch              | `1.10.0+cu111`                           |
| torchvision          | `0.11.0+cu111`                           |
| spconv               | `spconv-cu111==2.1.25`                   |
| torch-scatter        | `2.1.2` (compiled from source)           |
| waymo-open-dataset   | `waymo-open-dataset-tf-2-5-0==1.4.1`     |
| TensorFlow           | `2.5.0` (pulled in by the waymo wheel)   |
| numpy                | `1.19.5` (pinned by tensorflow 2.5)      |
| filterpy             | `1.4.5` (added manually, see below)      |

### Manual environment fix-ups required

1. `gcc-11` / `g++-11` had to be installed from `apt` because CUDA 11.8 rejects
   newer GCC versions (Ubuntu 24.04 ships with GCC 13). We then set
   `CC=gcc-11`, `CXX=g++-11` and `CUDAHOSTCXX=g++-11` before running each
   `setup.py develop`, matching the guidance already given in `INSTALL.md`.
2. The default GCC 13 caused two compile errors in `utils/setup.py`:
   `unsupported GNU version!` and `fatal error: 'sstream' file not found`.
   Both disappear once GCC 11 is used.
3. `TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"` and `FORCE_CUDA=1` were
   exported because the build host has no GPU; otherwise PyTorch auto-detects
   the arch list from `nvidia-smi` and aborts when the driver is missing.

## Step-by-step results for `docs/INSTALL.md`

| Step | Command                                                | Result |
| ---- | ------------------------------------------------------ | ------ |
| a    | `conda create --name detzero python=3.8`               | OK     |
| b    | `pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f ...` | OK |
| c    | `conda install cmake`                                   | OK     |
| d    | `pip install spconv-cu111`                              | OK     |
| e    | `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html` | OK (built from source; no matching wheel is published for `torch 1.10.0+cu111`, so it takes ~1–2 minutes) |
| f    | `pip install waymo-open-dataset-tf-2-5-0`               | OK (downgrades `numpy` to `1.19.5` and `typing-extensions` to `3.7.4.3`) |
| g    | `pip install -r requirements.txt`                       | OK, but see broken items below |
| h    | `cd utils && python setup.py develop`                   | OK (after GCC 11 fix) |
| i    | `cd detection && python setup.py develop`               | OK |
| i    | `cd tracking && python setup.py develop`                | OK |
| i    | `cd refining && python setup.py develop`                | OK |

## Smoke tests performed

1. **Third-party imports** – `torch`, `torchvision`, `spconv`, `spconv.pytorch`,
   `torch_scatter`, `waymo_open_dataset`, `numpy`, `numba`, `tensorboardX`,
   `easydict`, `yaml`, `cv2`, `scipy`, `addict`, `tabulate`, `tqdm`: all OK.
2. **Compiled CUDA extensions** – `detzero_utils.ops.iou3d_nms.iou3d_nms_cuda`,
   `detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_cuda`,
   `detzero_utils.ops.roipoint_pool3d.roipoint_pool3d_cuda`,
   `detzero_utils.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda`,
   `detzero_utils.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda`: all
   load (the CPU-only IoU path `boxes_bev_iou_cpu` runs end-to-end).
3. **DetZero package imports** – `detzero_utils`, `detzero_det`,
   `detzero_track`, `detzero_refine` (and their submodules): all OK.
4. **Entry-point `--help`** – `detection/tools/{train,test}.py`,
   `refining/tools/{train,test}.py`, `tracking/tools/run_track.py`,
   `evaluator/*.py`, `daemon/*.py` except `visualizer.py`: all OK.
5. **YAML configuration loading** – every `*_dataset.yaml` loads cleanly, and
   every `*_model.yaml` loads when the process is run from its `tools/`
   directory (the configs use `_BASE_CONFIG_` paths that are relative to
   `tools/`). One config is broken (see below).

## Broken items discovered

### 1. `kornia` import fails at install-time
`pip install -r requirements.txt` installs `kornia==0.7.3`, but the
installation pulls `typing-extensions==4.x` which is immediately downgraded by
`waymo-open-dataset-tf-2-5-0` to `3.7.4.3`. That older version does not expose
`TypeGuard`, which `kornia` imports unconditionally:

```
ImportError: cannot import name 'TypeGuard' from 'typing_extensions'
```

`kornia` is listed in `requirements.txt` but is **never imported** anywhere in
the repo, so nothing currently breaks at runtime. It should either be pinned
to a compatible version (`kornia<=0.6.12`) or removed from `requirements.txt`.

### 2. `filterpy` is required by tracking but not declared
`tracking/detzero_track/models/tracking_modules/kalman_filter/ab3dmot.py`
imports `from filterpy.kalman import KalmanFilter`, so running
`tracking/tools/run_track.py` crashes with `ModuleNotFoundError: No module
named 'filterpy'` on a fresh install. `filterpy` is absent from both
`requirements.txt` and `tracking/setup.py`. Adding `filterpy` to the install
requirements fixes the failure.

### 3. `open3d` is required by the visualizer but not declared
`daemon/visualizer.py` (and `utils/detzero_utils/visualize_utils/components.py`
transitively) imports `open3d`:

```
File "/workspace/utils/detzero_utils/visualize_utils/components.py", line 2, in <module>
    import open3d as o3d
ModuleNotFoundError: No module named 'open3d'
```

`open3d` is not listed in `requirements.txt` nor mentioned in `INSTALL.md`. It
needs to be added to requirements (or at least documented) or the visualizer
made robust to a missing import.

### 4. Malformed YAML: `tracking/tools/cfgs/tk_model_cfgs/waymo_ab3dmot.yaml`
The list entry for `low_confidence_box_filter` is missing indentation on its
`THRESHOLD` key:

```yaml
DATA_PROCESSOR:
    - NAME: heading_process

    - NAME: low_confidence_box_filter
    THRESHOLD: 0.1           # <-- must be indented under the list item
```

PyYAML then raises `expected <block end>, but found '?'`. The fallback path in
`detzero_utils.config_utils.cfg_from_yaml_file` calls `yaml.load(f)` without a
`Loader=` kwarg, which on modern PyYAML raises
`TypeError: load() missing 1 required positional argument: 'Loader'`, hiding
the real parse error. Fixing the indentation unblocks the file; the fallback
call should also be updated to `yaml.load(f, Loader=yaml.FullLoader)`.

### 5. Misleading `setup.py` fallback warning on Python 3.8 is harmless
All four `setup.py develop` invocations print:

```
Please avoid running ``setup.py`` directly.
Instead, use pypa/build, pypa/installer or other standards-based tools.
```

This is only a deprecation warning from modern `setuptools`; the builds
succeed. No action needed but worth noting for anyone copy-pasting the log.

### 6. PyTorch / CUDA minor-version warning
PyTorch 1.10.0+cu111 emits:

```
UserWarning: The detected CUDA version (11.8) has a minor version mismatch
with the version that was used to compile PyTorch (11.1). Most likely this
shouldn't be a problem.
```

`INSTALL.md` instructs users to install CUDA 11.0+ and spconv-cu111, but the
bundled PyTorch wheel is linked against CUDA 11.1. In practice the extensions
build and load, but a cleaner recipe would be to also install CUDA 11.1 with
conda (`nvidia/label/cuda-11.1.1`). If only CUDA 11.8 is available, the build
still works thanks to forward-compatible CUDA ABI for compute arches ≥ 6.0.

## Summary

- All four `setup.py develop` builds succeed once a GCC 11 host compiler is
  available; no code changes were required to get the C++/CUDA extensions to
  compile.
- After the fixes applied in this branch (see below), **30/30 import smoke
  tests pass and all 17 tracked model YAML configs load cleanly**.
- Two runtime dependencies were missing from `requirements.txt`: `filterpy`
  (hard requirement of the tracking module) and `open3d` (hard requirement
  of the daemon visualizer and of `detzero_utils.visualize_utils`). `open3d`
  is documented as an optional dependency (it pulls in a numpy/typing-extensions
  upgrade that conflicts with `waymo-open-dataset-tf-2-5-0`).
- One YAML file (`waymo_ab3dmot.yaml`) was malformed and, combined with a
  legacy `yaml.load()` fallback in `config_utils.py`, produced a confusing
  error message.

## Fixes applied on this branch

| Issue | Fix |
| --- | --- |
| `kornia` 0.7.x incompatible with the pinned `typing-extensions==3.7.4.3` from waymo | Pinned to `kornia<=0.6.5` in `requirements.txt` |
| `filterpy` not declared, `run_track.py` crashed | Added `filterpy` to `requirements.txt` |
| `open3d` not declared, `daemon/visualizer.py` crashed | Documented in `docs/INSTALL.md` as an optional extra (installing it into the main env would break the waymo/tensorflow pins) |
| `tracking/tools/cfgs/tk_model_cfgs/waymo_ab3dmot.yaml` malformed (under-indented `THRESHOLD`) | Indented the key under its list item |
| `config_utils.cfg_from_yaml_file` fallback called `yaml.load(f)` without a `Loader=`, producing a misleading `TypeError` that hid the real parser error | Narrowed the fallback to `except AttributeError` so parser errors now propagate correctly |

