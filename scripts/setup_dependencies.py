#!/usr/bin/env python3
"""Fetch the exact header dependencies recorded in the source workspace."""
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1] / "third_party"
    pins = json.loads((root / "versions.lock.json").read_text())
    for name, pin in pins.items():
        target = root / name
        if not target.exists():
            target.mkdir()
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", pin["url"]], check=True)
        remote = subprocess.check_output(["git", "-C", str(target), "remote", "get-url", "origin"], text=True).strip()
        if remote != pin["url"]:
            raise RuntimeError(f"Unexpected remote at {target}; preserve it and resolve manually.")
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if dirty:
            raise RuntimeError(f"Dependency has local changes: {target}")
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() == pin["commit"]:
            print(f"Already pinned: {name}")
            continue
        if head.returncode == 0:
            raise RuntimeError(f"Existing {target} uses another revision; resolve explicitly.")
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", pin["commit"]], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", pin["commit"]], check=True)


if __name__ == "__main__":
    main()
