#!/usr/bin/env python3
"""Recycle committed calibration workers; bounded retries for external SIGKILL."""
import argparse
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-recycles", type=int, default=100)
    parser.add_argument("--kill-retries", type=int, default=2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("worker command required")
    child = None
    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    recycled = killed = 0
    while True:
        child = subprocess.Popen(command)
        code = child.wait()
        if code == 75 and recycled < args.max_recycles:
            recycled += 1
            print(f"[supervisor] fresh worker after committed progress ({recycled})", flush=True)
            continue
        if code == -signal.SIGKILL and killed < args.kill_retries:
            killed += 1
            print(f"[supervisor] SIGKILL; resume last commit, bounded retry {killed}/{args.kill_retries}", flush=True)
            continue
        raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
