#!/usr/bin/env bash
# PostToolUse evidence gate: lint the file that was just edited and, for modules
# that are type-checked, re-check types. Must stay fast (~2 s).
# Exit 2 sends stderr back to the agent as a blocking error it has to fix.
set -uo pipefail

input=$(cat)
file=$(printf '%s' "$input" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
[ -z "$file" ] && exit 0
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
rel="${file#"$root"/}"
cd "$root" || exit 0

case "$rel" in
  .mypy-baseline.json)
    echo "BLOCKED: .mypy-baseline.json is only rewritten by humans (scripts/mypy_ratchet.py --update)." \
         "Fix the type errors instead of absorbing them into the baseline. Revert this edit." >&2
    exit 2 ;;
  data/*|*.pth|*.pkl|*.npy)
    echo "BLOCKED: datasets, checkpoints and cached arrays are out of scope for edits. Revert this edit." >&2
    exit 2 ;;
  *.py) ;;
  *) exit 0 ;;
esac

command -v ruff >/dev/null 2>&1 || exit 0
# Diff-aware: legacy files already carry findings, so only fail when the edit adds some.
rules=(--select E9,F63,F82,F401,F811 --output-format concise --quiet)
now=$(ruff check "${rules[@]}" "$rel" 2>&1)
head=$(git show "HEAD:$rel" 2>/dev/null | ruff check "${rules[@]}" --stdin-filename "$rel" - 2>&1)
out=""
if [ "$(printf '%s' "$now" | grep -c ': [EF][0-9]')" -gt "$(printf '%s' "$head" | grep -c ': [EF][0-9]')" ]; then
  out="$now"$'\n'"Your edit added lint findings to $rel (fix the new ones; do not add noqa)."
fi

# Type-checked modules (listed in pyproject.toml [tool.mypy].files): ratchet must hold.
if grep -qF "\"$rel\"" pyproject.toml && python3 -c 'import mypy' 2>/dev/null; then
  ratchet=$(python3 scripts/mypy_ratchet.py 2>&1) || out="$out"$'\n'"$ratchet"
fi

if [ -n "${out//[$'\n ']/}" ]; then
  printf '%s\n' "$out" >&2
  exit 2
fi
exit 0
