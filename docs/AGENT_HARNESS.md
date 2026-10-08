# Agent harness

Everything around a coding agent that isn't the model: the guides that steer it
before it acts and the sensors that let it correct itself afterwards. This page
maps what DetZero ships to that model. Conventions for contributors live in
[`AGENTS.md`](../AGENTS.md).

|                | Computational (deterministic, cheap) | Inferential (LLM) |
|----------------|--------------------------------------|-------------------|
| **Guides** (before) | Shape contracts in signatures (`detzero_utils/shape_types.py`), `TypedDict` schemas in each package's `structures.py` | `AGENTS.md`, `CLAUDE.md` |
| **Sensors** (after) | ruff, mypy ratchet, pytest + Hypothesis, runtime shape checks, semgrep house rules | (PR review bots, optional) |

## Shape contracts as types

Tensor shapes are part of the signature, written with
[jaxtyping](https://github.com/patrick-kidger/jaxtyping):

```python
@shape_checked
def boxes_to_corners_3d(boxes3d: BoxesND) -> Corners3D: ...          # (N, 7 + C) -> (N, 8, 3)

@shape_checked
def boxes_iou_normal(boxes_a: Float[Tensor, "N 4"],
                     boxes_b: Float[Tensor, "M 4"]) -> Float[Tensor, "N M"]: ...
```

* **Static**: mypy sees `Float[Tensor, ...]` as `Tensor`, so ordinary type errors are caught.
* **Runtime**: with `DETZERO_SHAPE_CHECK=1`, `@shape_checked` wraps the function in
  `jaxtyped(typechecker=beartype)`. Every call checks the dtype kind, fixed sizes, and that
  each named dimension binds to one size across all arguments and the return value. A
  transposed or mis-sliced tensor fails *at the call* with a message naming the dimension,
  instead of broadcasting silently. The flag is off by default, so training and inference
  pay nothing; `tests/conftest.py` turns it on.
* **Fallback**: without jaxtyping (e.g. the Python 3.8 runtime env), the markers degrade to
  `typing.Annotated` metadata, so annotated modules still import.

Annotated so far:

| Package | Modules |
|---|---|
| `detzero_utils` | `box_utils`, `common_utils` (geometry helpers), `ops/iou3d_nms/iou3d_nms_utils`, `ops/roiaware_pool3d/roiaware_pool3d_utils` |
| `detzero_det` | `utils/box_coder_utils`, `utils/loss_utils`, `utils/centernet_utils`, `utils/model_nms_utils` |
| `detzero_track` | `utils/transform_utils` |
| `detzero_refine` | `utils/data_utils` (geometry helpers) |

Model `forward` methods take whole `batch_dict`s, which are already typed as `TypedDict`s in
each package's `structures.py`. Their field docstrings record the shapes, and those are the
next candidates for contracts.

## Tests

`tests/` runs on CPU with no compiled extensions: `tests/conftest.py` puts the four
packages on `sys.path`, provides the `version.py` modules that `setup.py develop` would
generate, stubs the CUDA extensions (calling into a stub raises a clear error), and turns
on shape checking. `tests/test_shape_contracts.py` checks the enforcement itself.

Geometry and coder helpers are covered by Hypothesis property tests rather than
hand-picked examples: rotation round trips and norm preservation, `limit_period` range,
corner centroids and edge lengths, BEV IoU bounds, symmetry and self-IoU,
`ResidualCoder` encode→decode round trips, and pose-inverse round trips.

## What runs where

| Stage | Budget | What runs | Where |
|---|---|---|---|
| Per edit (agent) | ~2 s | diff-aware ruff on the edited file; mypy ratchet if it is a type-checked module; blocks edits to the baseline and data | `.claude/hooks/check.sh` (PostToolUse) |
| Per command (agent) | instant | deny `--no-verify`, ratchet `--update`, `DETZERO_SHAPE_CHECK=0`, force-push, recursive deletes | `.claude/hooks/deny.sh` (PreToolUse) |
| Pre-commit | <10 s | ruff on harness code, block checkpoint/data files | `.pre-commit-config.yaml` |
| Pre-push | <60 s | mypy ratchet, CPU pytest | `.pre-commit-config.yaml` |
| Pre-merge | minutes | all of the above + semgrep house rules on the diff | `.github/workflows/harness.yml` |

A rule written only in `AGENTS.md` is followed most of the time. A rule enforced by a hook
or by CI is followed every time.

## The mypy ratchet

Annotating a legacy module makes mypy check function bodies that were never type-checked,
and that surfaces legacy issues such as numpy and torch values reused in one variable or
implicit `Optional`. Fixing these means editing behaviour, so it is left to separate
changes. Instead, `scripts/mypy_ratchet.py` pins the per-file error count of the files
listed in `[tool.mypy].files` in `.mypy-baseline.json`:

* a file whose count goes **up** fails, and the output lists only that file's errors;
* a file whose count goes **down** tightens the baseline automatically (commit the update).

The shape-contract modules above are on the list. Their new signatures surfaced 51 legacy
errors in 9 files, now baselined, including a few undefined names (`lidar_to_image` in
`box_utils.boxes3d_to_boxes2d`, `min_radius` and `nms_post_max_size` in
`centernet_utils.decode_bbox_from_heatmap`'s disabled circle-NMS branch).

## Linter messages are prompts

Every custom check (semgrep rules in `.semgrep/house-rules.yml`, hook and ratchet messages)
is written for the agent about to fix it: what is wrong, how to fix it, which file shows the
pattern, and which wrong fix not to try (`Any`, blanket ignores, turning the check off).
