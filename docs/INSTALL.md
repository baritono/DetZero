# Installation

## Prerequisites
All the codes are tested in the environment:
- Linux (tested on Ubuntu 16.04/18.04/20.04/22.04)
- Python 3.10 (see note below about `waymo-open-dataset` and newer Python versions)
- PyTorch 1.10 or higher
- CUDA 11.0 or higher
- GCC 5.4+ (but also note the highest version of GCC supported by the CUDA version used, for compiling CUDA code with nvcc).
- [spconv v2.x](https://github.com/traveller59/spconv)

### A note on Python version and `waymo-open-dataset`

DetZero depends on Waymo's official evaluation library (`waymo-open-dataset-*`)
for Waymo-format metrics, submission, and `.tfrecord` preprocessing.

Waymo currently publishes two families of wheels on PyPI:

| Package | Latest version | Target env |
| --- | --- | --- |
| `waymo-open-dataset-tf-2-11-0` | `1.6.1` (Dec 2023) | TF 2.11 / Python 3.10 |
| `waymo-open-dataset-tf-2-12-0` | `1.6.7` (Apr 2025) | TF 2.13 / Python 3.10 |

The wheels are built as generic `py3-none-manylinux` but pin very strict
transitive versions (`numpy==1.23.5`, `tensorflow==2.13`, `pandas==1.5.3`,
`setuptools==67.6.0`, etc.) and are only classified for Python 3.10 on PyPI.
Installing the package under Python 3.11+ fails because those pinned
dependencies do not all ship Python 3.11 / 3.12 wheels. The Waymo team has
acknowledged this on the upstream tracker
([waymo-research/waymo-open-dataset#868](https://github.com/waymo-research/waymo-open-dataset/issues/868))
and as of 2025 still has not shipped a Python 3.11 build. Their public
recommendation, echoed by the maintainers on that thread, is:

> **Use Python 3.10** and install the Waymo wheel with `pip install ... --no-deps`,
> then install the parts of the dependency closure you actually need at versions
> that are compatible with your rest-of-stack.

That is what this project follows below. Once Waymo ships a Python 3.11/3.12
build we will relax this constraint; until then the recommended interpreter
for DetZero is **Python 3.10**.

### Example CUDA environment setup

If you use PyTorch 2.6.0+cu124, which requires CUDA 12.4, which, in turn, only
supports gcc 11.4 or older when compiling CUDA code, then you need to make sure
the correct versions of CUDA and gcc/g++ are used during the setup process. One
way is to set the environment variables, such as:
```shell
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDAHOSTCXX=/usr/bin/g++-11
```

## Recommended Steps
**a. Create a conda virtual environment.**
```shell
  conda create --name detzero python=3.10
  conda activate detzero
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.

**c. Install cmake.**
```shell
  conda install cmake
```

**d. Install sparse conv.**
```shell
  pip install spconv-cu111
```

**e. Install pytorch scatter (for DynamicVFE).**
We suggest to follow the instructions of [torch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package based on your own environment [version](https://data.pyg.org/whl/).


**f. Install the Waymo evaluation module.**

Following Waymo's own recommendation (see note above), install the wheel with
`--no-deps` and then supply the small subset of runtime dependencies that
DetZero actually uses at versions that line up with our other requirements:

```shell
  pip install waymo-open-dataset-tf-2-12-0==1.6.7 --no-deps
  pip install "tensorflow==2.13.*" "protobuf>=3.20,<4" "numpy==1.23.5"
```

If you prefer the older TF 2.11 line (e.g. you need `tensorflow_graphics`),
use the equivalent:

```shell
  pip install waymo-open-dataset-tf-2-11-0==1.6.1 --no-deps
  pip install "tensorflow==2.11.*" "protobuf>=3.20,<4" "numpy==1.21.5"
```

**g. Install other required dependent libraries.**
```shell
  cd DetZero && pip install -r requirements.txt
```

**h. Compile other libraries.**
```shell
  cd DetZero/utils && python setup.py develop
```

**i. Compile the libraries of specific algorithm modules.**
```shell
  cd DetZero/detection && python setup.py develop
```
```shell
  cd DetZero/tracking && python setup.py develop
```
```shell
  cd DetZero/refining && python setup.py develop
```
