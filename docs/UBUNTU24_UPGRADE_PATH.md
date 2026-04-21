# Ubuntu 24.04 Dependency Upgrade Path

This document maps DetZero's current dependency pins to versions that are
officially supported on Ubuntu 24.04, and highlights likely code changes.

## 1) Current state in this repo

Current install guidance (`docs/INSTALL.md`) and requirements are pinned to an
older stack:

- Python 3.8
- PyTorch 1.10 + CUDA 11.1 (`torch==1.10.0+cu111`)
- `spconv-cu111`
- `waymo-open-dataset-tf-2-5-0`
- `numpy<=1.19.5`, `numba==0.48.0`

Those pins are not aligned with a modern Ubuntu 24.04 environment.

## 2) Official compatibility matrices (key dependencies)

### Python (conda env)

- PyTorch current stable installs require **Python 3.10+** on the official
  "Get Started" page.
- PyTorch release matrix also shows supported Python ranges by release.

Implication for this repo:
- Move conda env target from `python=3.8` to **`python=3.10`** as baseline.

### Waymo Open Dataset library (official package line)

- Current repo uses `waymo-open-dataset-tf-2-5-0` (TensorFlow 2.5 era).
- Official Waymo package line has moved to e.g.
  `waymo-open-dataset-tf-2-12-0` (latest 1.6.7 at time of writing), with:
  - `tensorflow==2.13`
  - `numpy==1.23.5`
  - manylinux_2_24/_2_28 wheels
  - Python classifier focused on 3.10

Implication:
- Replace `waymo-open-dataset-tf-2-5-0` with
  **`waymo-open-dataset-tf-2-12-0`**.
- Keep a Waymo-capable env at Python 3.10.

### CUDA

- NVIDIA CUDA 12.6 and CUDA 12.8 installation guides explicitly list
  **Ubuntu 24.04** as supported.
- CUDA 12.6/12.8 host compiler support tables include GCC major versions up to
  14 (range: 6.x-14.x).

Implication:
- Use **CUDA 12.6 or 12.8** on Ubuntu 24.04 (12.6 is the safest bridge with
  PyTorch and current ecosystem wheels).

### PyTorch

- PyTorch release compatibility matrix (official `RELEASE.md`) shows:
  - 2.7 stable CUDA: 11.8 and **12.6**
  - 2.8/2.9/2.10+ increasingly centered on 12.6/12.8+
- Get Started install matrix for Linux includes cu126/cu128 options on stable.

Implication:
- Prefer **PyTorch 2.7 + cu126** as a practical Ubuntu 24.04 target.

### GCC

- Ubuntu for Developers GCC availability page lists Ubuntu 24.04 default as
  GCC 13, with GCC 9-14 available.
- CUDA 12.6/12.8 compiler policy supports GCC 13.

Implication:
- Ubuntu 24.04 default **GCC 13** is valid.
- Keep optional `g++-12`/`g++-13` fallback available for extension builds.

## 3) Recommended upgrade profiles

### Profile A (single-env, pragmatic baseline)

- Python: 3.10 (conda)
- CUDA toolkit: 12.6
- PyTorch: 2.7 (cu126 wheel)
- GCC/G++: 13
- Waymo: `waymo-open-dataset-tf-2-12-0`

Why: every key component is on an officially supported line for Ubuntu 24.04.

### Profile B (safer operational split, recommended)

Because this repo mixes PyTorch/CUDA custom ops and Waymo TensorFlow tooling,
use two conda envs:

1. `detzero-train` (training/inference)
   - Python 3.10
   - PyTorch 2.7 + cu126
   - CUDA 12.6
   - GCC 13
   - spconv/torch_scatter matching this torch+cuda combo

2. `detzero-waymo` (preprocess/eval/submit scripts)
   - Python 3.10
   - `waymo-open-dataset-tf-2-12-0` (pulls TF 2.13 + numpy 1.23.5 pin set)
   - CPU-only use is acceptable for metrics/preprocess workflows

Why: reduces dependency solver and shared CUDA runtime conflicts.

## 4) Potential repo code updates needed

These are the likely code-level changes when unpinning old NumPy/Python-era
assumptions:

1. Replace deprecated NumPy aliases:
   - `np.int`, `np.bool`, `np.float`, `np.str`
   - with builtins (`int`, `bool`, `float`, `str`) or explicit NumPy dtypes
     (`np.int64`, `np.bool_`, etc.).

2. Candidate files already using deprecated aliases include:
   - `evaluator/waymo_eval_tracking.py`
   - `refining/detzero_refine/utils/data_utils.py`
   - `refining/detzero_refine/datasets/waymo/waymo_confidence_dataset.py`
   - `utils/detzero_utils/visualize_utils/components.py`
   - `utils/detzero_utils/box_utils.py`
   - `detection/detzero_det/models/centerpoint_modules/backbone2d.py`
   - `tracking/detzero_track/utils/track_calculation.py`
   - `tracking/detzero_track/utils/data_utils.py`
   - `tracking/detzero_track/models/tracking_modules/track_manager.py`
   - `tracking/detzero_track/models/tracking_modules/target_assign.py`
   - `tracking/detzero_track/models/tracking_modules/data_association/*.py`
   - `daemon/prepare_object_data.py`

3. Split requirements by purpose (recommended):
   - training/runtime requirements
   - waymo-eval/preprocess requirements

4. Update install docs to avoid hard-pinned legacy wheels (`cu111`, TF2.5).

## 5) Suggested execution order

1. Update docs + environment files first (no code behavior change).
2. Create new conda env(s) on Python 3.10.
3. Install PyTorch/CUDA stack and rebuild local CUDA extensions.
4. Install Waymo replacement package and run Waymo preprocess/eval smoke tests.
5. Apply NumPy alias cleanup across codebase.
6. Run end-to-end smoke tests for:
   - preprocess (`waymo_preprocess`)
   - detection training bootstrap
   - tracking + evaluator scripts

## 6) Official references used

- PyTorch Get Started: https://pytorch.org/get-started/locally/
- PyTorch release compatibility matrix:
  https://raw.githubusercontent.com/pytorch/pytorch/main/RELEASE.md
- CUDA 12.6 Linux install guide:
  https://docs.nvidia.com/cuda/archive/12.6.1/cuda-installation-guide-linux/index.html
- CUDA 12.8 Linux install guide:
  https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html
- Waymo package (current line):
  https://pypi.org/project/waymo-open-dataset-tf-2-12-0/
- Waymo package JSON metadata:
  https://pypi.org/pypi/waymo-open-dataset-tf-2-12-0/json
- Legacy Waymo package used by this repo:
  https://pypi.org/pypi/waymo-open-dataset-tf-2-5-0/json
- Ubuntu GCC availability:
  https://documentation.ubuntu.com/ubuntu-for-developers/reference/availability/gcc/
