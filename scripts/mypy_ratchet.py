#!/usr/bin/env python3
"""mypy ratchet: legacy errors are baselined; new ones fail.

Annotating a legacy module makes mypy check function bodies that were never
type-checked before, which surfaces pre-existing issues (numpy/torch variable
reuse, implicit Optional, ...).  Fixing all of them at once would mean
behaviour-touching edits to legacy code, so this script instead pins the
current per-file error count in ``.mypy-baseline.json`` and fails when any
file's count goes *up*.  When a count goes down, the baseline is tightened.

Usage:
    python scripts/mypy_ratchet.py            # check (CI / pre-push / agent hook)
    python scripts/mypy_ratchet.py --update   # rewrite the baseline (humans only)
"""

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / ".mypy-baseline.json"
ERROR_RE = re.compile(r"^(?P<file>[^:\s]+\.py):\d+: error: ")


def run_mypy() -> "tuple[Counter, str]":
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--show-error-codes"],
        cwd=ROOT, capture_output=True, text=True,
    )
    counts: Counter = Counter()
    for line in proc.stdout.splitlines():
        m = ERROR_RE.match(line)
        if m:
            counts[m.group("file")] += 1
    if proc.returncode not in (0, 1):  # 2 = mypy crashed / bad config
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit(proc.returncode)
    return counts, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update", action="store_true", help="rewrite the baseline to the current counts")
    args = parser.parse_args()

    counts, output = run_mypy()

    if args.update:
        BASELINE.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
        print(f"baseline updated: {sum(counts.values())} errors in {len(counts)} files")
        return 0

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    regressions = {f: (baseline.get(f, 0), n) for f, n in counts.items() if n > baseline.get(f, 0)}
    improved = {f: (n, counts.get(f, 0)) for f, n in baseline.items() if counts.get(f, 0) < n}

    if regressions:
        for line in output.splitlines():
            if line.split(":", 1)[0] in regressions:
                print(line)
        print("\nMYPY RATCHET FAILED: these files gained type errors:")
        for f, (old, new) in sorted(regressions.items()):
            print(f"  {f}: {old} -> {new}")
        print(
            "TO FIX: fix the new errors listed above in the files you changed.\n"
            "Do NOT widen an annotation to Any, do NOT add a code-less type-ignore comment, and do NOT run\n"
            "`--update` to absorb the regression. If the checker is genuinely wrong, use\n"
            "`# type: ignore[<code>]` with a one-line reason."
        )
        return 1

    if improved:
        BASELINE.write_text(json.dumps(dict(sorted((f, n) for f, n in counts.items() if n)), indent=2) + "\n")
        for f, (old, new) in sorted(improved.items()):
            print(f"ratchet tightened: {f}: {old} -> {new}")
        print("Commit the updated .mypy-baseline.json.")

    print(f"mypy ratchet OK ({sum(counts.values())} baselined errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
