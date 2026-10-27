"""Memory guard: run one command at a time under RAM and swap limits.

Every command that loads model weights runs under this guard, per the plan's
Global Constraints:

* an exclusive lock on ``$MAYA_FAST_WORK/model.lock`` (default
  ``~/.cache/maya_fast/model.lock``) is held for the whole run, so only one
  model process exists at a time across agents;
* the run is refused when free memory is below ``min_free_mb``;
* the child tree is killed when its RSS exceeds ``rss_limit_mb`` or when swap
  grew by more than ``max_swap_growth_mb`` during the run.

CLI:
    python tools/memguard.py --rss-limit-mb N --min-free-mb M -- <cmd...>
prints one line ``MEMGUARD peak_rss_mb=... swap_delta_mb=... reason=...``
and exits non-zero on any violation.

MB means MiB (1024**2) throughout.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

MB = 1024 * 1024


@dataclass
class GuardResult:
    returncode: int
    peak_rss_mb: float
    swap_before_mb: float
    swap_after_mb: float
    killed: bool
    reason: str | None


def lock_path() -> Path:
    """Shared model lock: one guarded model run at a time, across processes."""
    work = os.environ.get("MAYA_FAST_WORK") or str(Path.home() / ".cache" / "maya_fast")
    return Path(work) / "model.lock"


def _acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        print(f"MEMGUARD waiting for model lock {path}", file=sys.stderr, flush=True)
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    return fh


def _tree_rss(proc: subprocess.Popen) -> int:
    try:
        ps_proc = psutil.Process(proc.pid)
        processes = [ps_proc] + ps_proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    rss = 0
    for p in processes:
        try:
            rss += p.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return rss


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        ps_proc = psutil.Process(proc.pid)
        for p in ps_proc.children(recursive=True) + [ps_proc]:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - children ignored SIGKILL
        pass


def run_guarded(
    cmd: list[str],
    *,
    rss_limit_mb: int,
    min_free_mb: int,
    max_swap_growth_mb: int = 64,
    poll_s: float = 0.2,
) -> GuardResult:
    """Run ``cmd`` while holding the shared model lock and enforcing limits.

    The lock is taken *before* the free-memory check and held until the child
    exits, so a refused run still serializes against other model runs and a
    finished run frees its memory before the next one is checked.
    """
    lock = _acquire_lock(lock_path())
    try:
        swap_before = psutil.swap_memory().used
        if psutil.virtual_memory().available < min_free_mb * MB:
            return GuardResult(
                returncode=-1,
                peak_rss_mb=0.0,
                swap_before_mb=swap_before / MB,
                swap_after_mb=swap_before / MB,
                killed=False,
                reason="insufficient free memory",
            )

        proc = subprocess.Popen(cmd)
        peak_rss = 0
        killed = False
        reason: str | None = None
        while proc.poll() is None:
            rss = _tree_rss(proc)
            peak_rss = max(peak_rss, rss)
            if rss > rss_limit_mb * MB:
                reason, killed = "rss limit", True
                _kill_tree(proc)
                break
            if psutil.swap_memory().used - swap_before > max_swap_growth_mb * MB:
                reason, killed = "swap growth", True
                _kill_tree(proc)
                break
            time.sleep(poll_s)

        swap_after = psutil.swap_memory().used
        returncode = proc.returncode if proc.returncode is not None else -1
        if reason is not None and returncode == 0:
            returncode = -1
        return GuardResult(
            returncode=returncode,
            peak_rss_mb=peak_rss / MB,
            swap_before_mb=swap_before / MB,
            swap_after_mb=swap_after / MB,
            killed=killed,
            reason=reason,
        )
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        flag_args, cmd = argv[:split], argv[split + 1 :]
    else:
        flag_args, cmd = argv, []
    parser = argparse.ArgumentParser(prog="memguard", add_help=True)
    parser.add_argument("--rss-limit-mb", type=int, required=True)
    parser.add_argument("--min-free-mb", type=int, required=True)
    parser.add_argument("--max-swap-growth-mb", type=int, default=64)
    parser.add_argument("--poll-s", type=float, default=0.2)
    args = parser.parse_args(flag_args)
    if not cmd:
        parser.error("no command given; use: -- <cmd...>")

    result = run_guarded(
        cmd,
        rss_limit_mb=args.rss_limit_mb,
        min_free_mb=args.min_free_mb,
        max_swap_growth_mb=args.max_swap_growth_mb,
        poll_s=args.poll_s,
    )
    print(
        f"MEMGUARD peak_rss_mb={result.peak_rss_mb:.1f} "
        f"swap_delta_mb={result.swap_after_mb - result.swap_before_mb:.1f} "
        f"reason={result.reason}",
        flush=True,
    )
    if result.reason is not None:
        return 1
    return result.returncode if result.returncode >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())