#!/usr/bin/env bash
# PreToolUse permission gate for Bash. Deterministic, short, auditable.
set -uo pipefail

cmd=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)
deny() { echo "BLOCKED: $1" >&2; exit 2; }

case "$cmd" in
  *"--no-verify"*)               deny "bypassing hooks is never the fix. Fix the failure the hook reports." ;;
  *"mypy_ratchet.py --update"*)  deny "the mypy baseline is only rewritten by a human. Fix the new type errors instead." ;;
  *"DETZERO_SHAPE_CHECK=0"*)     deny "do not disable shape checking to get a test to pass; fix the shape mismatch it reports." ;;
  *"push --force"*|*"push -f"*)  deny "force-pushing rewrites shared history. Ask the human." ;;
  *"rm -rf data"*|*"rm -rf /"*)  deny "no recursive deletes of data or system paths. Run 'git clean -n' and ask." ;;
esac
exit 0
