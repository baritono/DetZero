# Agent harness

Everything around a coding agent that isn't the model: the guides that steer it
before it acts and the sensors that let it correct itself afterwards. This page
maps what DetZero ships to that model. Conventions for contributors live in
[`AGENTS.md`](../AGENTS.md).

|                | Computational (deterministic, cheap) | Inferential (LLM) |
|----------------|--------------------------------------|-------------------|
| **Guides** (before) | Type annotations, `TypedDict` schemas in each package's `structures.py` | `AGENTS.md`, `CLAUDE.md` |
| **Sensors** (after) | ruff, mypy ratchet, pytest + Hypothesis, semgrep house rules | (PR review bots, optional) |

## Tests

`tests/` runs on CPU with no compiled extensions: `tests/conftest.py` puts the four
packages on `sys.path`, provides the `version.py` modules that `setup.py develop` would
generate, and stubs the CUDA extensions (calling into a stub raises a clear error).

Geometry and coder helpers are covered by Hypothesis property tests rather than
hand-picked examples: rotation round trips and norm preservation, `limit_period` range,
corner centroids and edge lengths, BEV IoU bounds, symmetry and self-IoU,
`ResidualCoder` encode→decode round trips, and pose-inverse round trips.

## What runs where

| Stage | Budget | What runs | Where |
|---|---|---|---|
| Per edit (agent) | ~2 s | diff-aware ruff on the edited file; mypy ratchet if it is a type-checked module; blocks edits to the baseline and data | `.claude/hooks/check.sh` (PostToolUse) |
| Per command (agent) | instant | deny `--no-verify`, ratchet `--update`, force-push, recursive deletes | `.claude/hooks/deny.sh` (PreToolUse) |
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

Today the list covers `tests/` and `scripts/`, which are clean, so no baseline file exists
yet.

## Linter messages are prompts

Every custom check (semgrep rules in `.semgrep/house-rules.yml`, hook and ratchet messages)
is written for the agent about to fix it: what is wrong, how to fix it, which file shows the
pattern, and which wrong fix not to try (`Any`, blanket ignores, turning the check off).
