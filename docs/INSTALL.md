# Installation

## Prerequisites
All the code is tested in the following environment:
- Linux (tested on Ubuntu 20.04 / 22.04 / **24.04 LTS**)
- Python 3.10 (required by `waymo-open-dataset-tf-2-12-0`, see the version matrix below)
- PyTorch 2.6.0 (or newer) built against CUDA 12.6
- CUDA Toolkit 12.6 (first officially supported CUDA on Ubuntu 24.04 is 12.5.1; 12.6 is recommended)
- gcc / g++ 13 (Ubuntu 24.04 default; within the CUDA 12.6 host-compiler support range)
- [spconv v2.x](https://github.com/traveller59/spconv)

### Compatibility matrix (Ubuntu 24.04)

| Dependency | Required version | Rationale |
|---|---|---|
| Ubuntu | 24.04 LTS (Noble Numbat) | Default gcc-13 (13.3.0), glibc 2.39, kernel 6.8 |
| Python | **3.10** | `waymo-open-dataset-tf-2-12-0==1.6.7` is classified `Python :: 3.10` and hard-pins `numpy==1.23.5` / `dask==2023.3.1`, which currently breaks under Python 3.11/3.12 |
| CUDA Toolkit | **12.6** (12.5.1 minimum) | [NVIDIA CUDA 12.6 installation guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-installation-guide-linux/) lists Ubuntu 24.04 as a supported distro; CUDA 12.4 does **not** officially support 24.04 |
| gcc / g++ | **13** (Ubuntu 24.04 default) | CUDA 12.6 host-compiler policy accepts GCC up to 13.2; NVCC only checks the major version so gcc-13.3 from Ubuntu 24.04 works in practice |
| PyTorch | **2.6.0+cu126** (or 2.7+) | PyTorch 2.6 supports Python 3.9–3.13 and ships a CUDA 12.6 wheel (experimental in 2.6, stable from 2.7) |
| torchvision | 0.21.0 (for torch 2.6) | Matches torch 2.6 release |
| spconv | **`spconv-cu126==2.3.8`** | Matches CUDA 12.6, requires Python ≥ 3.9 |
| waymo-open-dataset | **`waymo-open-dataset-tf-2-12-0==1.6.7`** | Latest official package (Apr 2025); bundles TensorFlow 2.12, numpy 1.23.5 |
| numpy | **1.23.5** | Hard-pinned by `waymo-open-dataset-tf-2-12-0` |
| numba | **>=0.59,<0.60** | First Numba release with Python 3.12 support is 0.59, still compatible with numpy 1.23 |

Make sure that you set the software environment correctly before running `setup.py`.

PyTorch 2.6.0+cu126 requires CUDA 12.6, which, in turn, requires gcc 13 or older when compiling
CUDA code. Ubuntu 24.04 provides `gcc-13` / `g++-13` as the default; if a newer gcc has been
installed system-wide, point nvcc at gcc-13 explicitly:
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
sudo apt-get install -y build-essential gcc-13 g++-13
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
pip install waymo-open-dataset-tf-2-12-0==1.6.7
```
This upgrade (from the previous `waymo-open-dataset-tf-2-5-0`) pins `numpy==1.23.5`, `tensorflow==2.12`,
`pandas==1.5.3` and related dependencies automatically.

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
