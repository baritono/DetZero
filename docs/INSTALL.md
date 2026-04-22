# Installation

## Prerequisites
All the code is tested in the following environment:
- Linux (tested on Ubuntu 20.04 / 22.04 / **24.04 LTS**)
- Python 3.10 (required by `waymo-open-dataset-tf-2-12-0`, see the version matrix below)
- PyTorch 2.6.0 built against CUDA 12.6
- CUDA Toolkit 12.6 (first officially supported CUDA on Ubuntu 24.04 is 12.5.1; 12.6 is recommended)
- gcc / g++ 13 (Ubuntu 24.04 default; within the CUDA 12.6 host-compiler support range)
- [spconv v2.x](https://github.com/traveller59/spconv)

### Compatibility matrix (Ubuntu 24.04)

| Dependency | Required version | Rationale |
|---|---|---|
| Ubuntu | 24.04 LTS (Noble Numbat) | Default gcc-13 (13.3.0), glibc 2.39, kernel 6.8 |
| Python | **3.10** | `waymo-open-dataset-tf-2-12-0` is classified `Python :: 3.10` on PyPI. Its hard-pinned `numpy==1.23` / `dask==2023.3.1` currently break installation on Python 3.11+ (see [waymo-open-dataset#868](https://github.com/waymo-research/waymo-open-dataset/issues/868)). Ubuntu 24.04 ships Python 3.12 by default; 3.10 is available through the [`deadsnakes` PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) or via `conda create python=3.10`. |
| CUDA Toolkit | **12.6** (12.5.1 minimum) | The [NVIDIA CUDA 12.6 installation guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-linux/) lists Ubuntu 24.04 as a supported distro; CUDA 12.4 does **not** officially support 24.04. |
| gcc / g++ | **13** (Ubuntu 24.04 default) | The CUDA 12.6 host-compiler policy accepts GCC up to 13.2; NVCC only checks the major version, so Ubuntu 24.04's gcc-13.3 works in practice. |
| PyTorch | **2.6.0+cu126** | PyTorch 2.6 supports Python 3.9–3.13. Upstream doesn't certify individual Linux distros; it ships manylinux wheels. The CUDA 12.6 wheel is built against **manylinux 2.28** (glibc ≥ 2.28), and Ubuntu 24.04 has glibc 2.39, so the wheel loads cleanly. (CUDA 11.8 / 12.4 torch 2.6 wheels are still manylinux2014 / glibc ≥ 2.17, also fine on 24.04.) |
| torchvision | 0.21.0 (for torch 2.6) | Matches torch 2.6 release. |
| spconv | **`spconv-cu126==2.3.8`** | Matches CUDA 12.6, requires Python ≥ 3.9. |
| waymo-open-dataset | **`waymo-open-dataset-tf-2-12-0==1.6.4`** | See the note below — this is the last waymo release whose transitive dependencies resolve cleanly against PyTorch 2.6. |
| numpy | 1.23.x (hard-pinned by waymo) | Do not add a conflicting pin to `requirements.txt`; let the waymo package pick the version. |
| numba | **>=0.59,<0.60** | First Numba release with Python 3.12 support; still compatible with numpy 1.23 (numba 0.61 drops numpy < 1.24). |
| typing-extensions | >=4.10.0 | Pulled in transitively by torch 2.6. |

### Why not the latest waymo pip package?

`waymo-open-dataset-tf-2-12-0` versions **1.6.5, 1.6.6 and 1.6.7** all pin `tensorflow==2.13` in
their published metadata. TensorFlow 2.13 in turn hard-pins `typing-extensions<4.6.0,>=3.6.6`
([TF issue #61848](https://github.com/tensorflow/tensorflow/issues/61848)). PyTorch 2.6.0 requires
`typing-extensions>=4.10.0` ([torch PR #133887](https://github.com/pytorch/pytorch/pull/133887)),
so pip's resolver errors out with:
```
ResolutionImpossible: typing-extensions<4.6.0,>=4.10.0
```

`waymo-open-dataset-tf-2-12-0==1.6.4` (the previous release) instead pins `tensorflow==2.12`,
whose `typing-extensions>=3.6.6` has **no** upper bound, so it coexists with torch 2.6.0+ and
resolves cleanly.

If a newer waymo-open-dataset release ever relaxes the TF 2.13 pin (or upgrades past TF 2.14,
which also drops the typing-extensions cap), this pin can be loosened accordingly.

### Environment variables for nvcc

PyTorch 2.6.0+cu126 requires CUDA 12.6, whose host-compiler policy caps GCC at major version 13.
Ubuntu 24.04 provides `gcc-13` / `g++-13` as the default. If a newer gcc has been installed
system-wide, point nvcc at gcc-13 explicitly:
```shell
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDAHOSTCXX=/usr/bin/g++-13
```

## Recommended Steps

**a. Install the Ubuntu 24.04 toolchain (once per machine).**
```shell
sudo apt-get update
# build-essential brings gcc-13/g++-13 as defaults on 24.04; OpenEXR headers
# are required by `openexr==1.3.9`, a transitive dependency of waymo-open-dataset 1.6.4.
sudo apt-get install -y build-essential gcc-13 g++-13 libopenexr-dev
# Install CUDA Toolkit 12.6 from https://developer.nvidia.com/cuda-12-6-0-download-archive
# choosing Ubuntu 24.04 as the target distribution.
```

**b. Create a conda virtual environment with Python 3.10.**
```shell
conda create --name detzero python=3.10
conda activate detzero
```
(If you prefer not to use conda, a `python3.10 -m venv` environment from the
[deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) works equivalently.)

**c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126
```
Note: make sure that your compilation CUDA version and runtime CUDA version match.

**d. Install cmake.**
```shell
conda install cmake
```

**e. Install sparse conv (matching CUDA 12.6).**
```shell
pip install spconv-cu126==2.3.8
```

**f. Install pytorch-scatter (for DynamicVFE).**
Follow the instructions of [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) to pick
a wheel matching your torch / CUDA version, for example:
```shell
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```

**g. Install the Waymo evaluation module.**
```shell
pip install waymo-open-dataset-tf-2-12-0==1.6.4
```
This pulls in `tensorflow==2.12`, `numpy==1.23`, `pandas==1.5.3`, `dask==2023.3.1` and other
transitive deps automatically. Do **not** upgrade to 1.6.5+ while staying on PyTorch 2.6 — see the
"Why not the latest waymo pip package?" note above.

**h. Install other required dependent libraries.**
```shell
cd DetZero && pip install -r requirements.txt
```

**i. Compile DetZero's own CUDA / C++ extensions.**
```shell
cd DetZero/utils && python setup.py develop
```
```shell
cd DetZero/detection && python setup.py develop
```
```shell
cd DetZero/tracking && python setup.py develop
```
```shell
cd DetZero/refining && python setup.py develop
```

### Verifying a resolvable environment

After step `g` completes without a `ResolutionImpossible` error from pip, you can additionally
sanity-check the resolved versions with:
```shell
pip install pipdeptree
pipdeptree -p torch -p tensorflow -p numpy -p typing_extensions
```
Expected outputs are `torch==2.6.0`, `tensorflow==2.12.x`, `numpy==1.23.x`, and
`typing_extensions>=4.10.0,<5`.
