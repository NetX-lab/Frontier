#!/usr/bin/env python3
"""Measure one commit against the captured baseline, from a detached checkout.

    PYTHONPATH=$PWD python tests/e2e/refactor_fidelity/measure_commit.py \\
        --sha 99922d2 --output-root "$FRONTIER_TMP_ROOT/refactor-fidelity"

Running the matrix against a shared working tree measures whatever happens to
be on disk at that moment.  When two people work on one branch, that is a
mixture, and a mixture is not a verdict on anything: one such run reported 66
failures that belonged to somebody else's half-finished edit.

This driver removes that failure mode. It checks the commit out into its own
detached worktree, runs the matrix there under a label named after the commit,
and compares against the baseline captured earlier.  The harness it runs is the
one committed at that revision, so the measurement and the code being measured
always agree.

The worktree is kept by default: a validated checkout of an earlier split is
the reference you want when a later split reports a difference and you need to
attribute it.  Pass --remove-worktree when you no longer need it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(cwd) if cwd else None,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _main_worktree(start: Path) -> Path:
    """Return the repository's main worktree, given any worktree inside it."""

    common_dir = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=start))
    return common_dir.parent


def _run(command: Sequence[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=str(cwd), check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sha", required=True, help="commit to measure")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--label", default=None,
                        help="defaults to candidate_<short sha>")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="measure only the first N cases, for a plumbing check")
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--remove-worktree", action="store_true",
                        help="delete the detached checkout after comparing")
    args = parser.parse_args(argv)

    here = Path.cwd()
    main_worktree = _main_worktree(here)
    full_sha = _git("rev-parse", f"{args.sha}^{{commit}}", cwd=here)
    short_sha = full_sha[:7]
    label = args.label or f"candidate_{short_sha}"
    checkout = main_worktree / ".worktrees" / f"fidelity-candidate-{short_sha}"

    print(f"commit   {full_sha}")
    print(f"subject  {_git('log', '-1', '--format=%s', full_sha, cwd=here)}")
    print(f"label    {label}")
    print(f"checkout {checkout}")

    if checkout.exists():
        existing = _git("rev-parse", "HEAD", cwd=checkout)
        if existing != full_sha:
            print(
                f"refusing to reuse {checkout}: it is at {existing}, not {full_sha}",
                file=sys.stderr,
            )
            return 2
        print("reusing the existing detached checkout at the same commit")
    else:
        _git("worktree", "add", "--detach", str(checkout), full_sha, cwd=main_worktree)

    driver = checkout / "tests" / "e2e" / "refactor_fidelity" / "run_matrix.py"
    if not driver.is_file():
        print(f"the harness is not present at {full_sha}: {driver}", file=sys.stderr)
        return 2

    run_command = [
        args.python_bin, str(driver), "run",
        "--repo-root", str(checkout),
        "--label", label,
        "--output-root", args.output_root,
        "--python-bin", args.python_bin,
        "--jobs", str(args.jobs),
        "--clean-cache",
        "--continue-on-failure",
    ]
    if args.limit is not None:
        run_command += ["--limit", str(args.limit)]
    if args.case_filter:
        run_command += ["--case-filter", args.case_filter]

    # The driver imports tests.e2e.refactor_fidelity, so it needs its own
    # checkout on the path, not the one this script was invoked from.
    import os
    env_note = f"PYTHONPATH={checkout}"
    print(f"\n({env_note})")
    os.environ["PYTHONPATH"] = str(checkout)

    if _run(run_command, cwd=checkout) != 0:
        print("\nthe run reported failing cases; comparing anyway", file=sys.stderr)

    compare_command = [
        args.python_bin, str(driver), "compare",
        "--output-root", args.output_root,
        "--baseline-label", args.baseline_label,
        "--candidate-label", label,
    ]
    verdict = _run(compare_command, cwd=checkout)

    if args.remove_worktree:
        _git("worktree", "remove", "--force", str(checkout), cwd=main_worktree)
        print(f"removed {checkout}")
    else:
        print(f"\nkept {checkout} as a reference; "
              f"remove it with: git -C {main_worktree} worktree remove {checkout}")

    print("\nVERDICT: " + ("IDENTICAL" if verdict == 0 else "DIFFERENCES FOUND"))
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
