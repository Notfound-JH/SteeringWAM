"""Periodically push committed recovery code; never stage or commit the worktree."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent


def git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "credential.interactive=never", *args],
            cwd=ROOT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"git {args[0]} timed out after 120 seconds") from error
    if result.returncode:
        # Git errors may echo credential-bearing URLs; report operation and exit
        # status without copying them into episode data.
        raise RuntimeError(f"git {args[0]} failed with exit code {result.returncode}; check GitHub authentication and branch permissions")
    return result.stdout.strip()


def sync_once(branch: str) -> dict:
    ref = f"refs/heads/{branch}"
    commit = git("rev-parse", "--verify", ref)
    git("push", "--porcelain", "origin", f"{commit}:{ref}")
    remote = git("ls-remote", "--exit-code", "origin", ref).split()[0]
    if remote != commit:
        raise RuntimeError("Remote branch does not match the committed recovery checkpoint")
    return {"status": "synced", "branch": branch, "commit": commit, "remote_commit": remote}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", default="feature/wam-agent-recovery")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".blob-sync/recovery-code")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("interval must be finite and at least one second")
    git("check-ref-format", "--branch", args.branch)
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    with (args.state_dir / "sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stopped.is_set():
            try:
                status = sync_once(args.branch)
            except RuntimeError as error:
                status = {"status": "blocked", "branch": args.branch, "error": str(error)}
            status["checked_at"] = time.time()
            temporary = args.state_dir / "status.tmp"
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(args.state_dir / "status.json")
            print(json.dumps(status), flush=True)
            if args.once:
                return 0 if status["status"] == "synced" else 1
            stopped.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
