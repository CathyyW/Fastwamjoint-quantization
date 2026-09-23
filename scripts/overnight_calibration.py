#!/usr/bin/env python3
"""Fail-closed offline supervisor. Default: read-only check; --run opts in."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


class Blocked(RuntimeError):
    pass


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_plan(path):
    path = Path(path).expanduser().resolve()
    plan = json.loads(path.read_text())
    if plan.get("mode", "full") not in ("full", "smoke_only"):
        raise ValueError("mode must be full or smoke_only")
    def resolve(value):
        return str((path.parent / Path(value).expanduser()).resolve())
    plan["output_dir"] = resolve(plan["output_dir"])
    for task in plan["tasks"]:
        task["config"] = resolve(task["config"])
    for item in plan["downloads"]:
        item["path"] = resolve(item["path"])
        if item["bytes"] <= 0 or len(item["sha256"]) != 64:
            raise ValueError("Each download needs pinned bytes/SHA256.")
    names = [task["name"] for task in plan["tasks"]]
    if not names or len(set(names)) != len(names) or any(n not in ("pack", "stack") for n in names):
        raise ValueError("Use distinct pack/stack task names.")
    paths = [item["path"] for item in plan["downloads"]]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Download paths must be nonempty and unique.")
    for key in ("poll_seconds", "download_timeout_hours", "download_idle_minutes", "stage_timeout_hours"):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 1 <= plan["poll_seconds"] <= 30:
        raise ValueError("poll_seconds must be in [1,30].")
    for value in plan["disk_gib"].values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Disk budgets must be positive finite GiB values.")
    if str(plan["gpu"]) not in ("0", "1", "2", "3", "4", "5", "6", "7"):
        raise ValueError("Choose one GPU index.")
    return plan


def prerequisites(plan):
    """Fast, read-only diagnostics; actual adapter checks run in a child too."""
    missing = []
    for task in plan["tasks"]:
        config = json.loads(Path(task["config"]).read_text())
        label = task["name"]
        if config.get("real_robot", {}).get("task") != label:
            missing.append(f"{label}: config task mismatch")
        for key in ("stats_checkpoint_binding", "training_split", "image_pipeline", "vae_identity"):
            value = config.get("calibration_approval", {}).get(key)
            if not isinstance(value, str) or not value.strip():
                missing.append(f"{label}: calibration_approval.{key}")
        scale = config.get("normalized_action_scale", {})
        values = scale.get("values", [])
        if (not isinstance(scale.get("evidence"), str) or not scale["evidence"].strip()
                or len(values) != 14 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in values)):
            missing.append(f"{label}: normalized_action_scale (14 positive values + evidence)")
        manifest = config.get("dataset", {}).get("sampling_manifest")
        if not manifest or not Path(manifest).is_absolute() or not Path(manifest).is_file():
            missing.append(f"{label}: reviewed absolute dataset.sampling_manifest path")
    return missing


def fingerprint(plan):
    files = [p for p in (REPO / "src").rglob("*") if p.suffix in (".py", ".cu", ".cpp", ".cuh", ".h")]
    files += list((REPO / "scripts").glob("*.py"))
    files += list((REPO / "tests").glob("*.py"))
    if plan.get("plan_path"):
        files.append(Path(plan["plan_path"]))
    for task in plan["tasks"]:
        path = Path(task["config"])
        files.append(path)
        selection = json.loads(path.read_text()).get("dataset", {}).get("sampling_manifest")
        if selection and Path(selection).is_file():
            files.append(Path(selection))
    value = {"plan": plan, "python": sys.executable,
             "files": {str(p.resolve()): digest(p) for p in sorted(set(files))}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def download_snapshot(plan):
    missing, progress = [], []
    for item in plan["downloads"]:
        final = Path(item["path"])
        partial = final.with_name(final.name + ".part")
        current = final if final.exists() else partial
        size = current.stat().st_size if current.exists() else 0
        if size > item["bytes"] or (final.exists() and size != item["bytes"]):
            raise Blocked(f"Wrong download size; file retained: {current}")
        if not final.is_file():
            missing.append(str(final))
        progress.append((str(current), size))
    return missing, progress


def stop_group(process):
    # Only the child process group created by this supervisor; not download jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader; descendants may still be shutting down.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


class Supervisor:
    def __init__(self, plan):
        self.plan = plan
        self.root = Path(plan["output_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = fingerprint(plan)
        self.env = os.environ.copy()
        self.env["PYTHONPATH"] = str(REPO / "src")
        self.env["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
        self.env.setdefault("OMP_NUM_THREADS", "4")
        self.env.setdefault("MAX_JOBS", "4")
        self.stage = "initializing"

    def status(self, state, **info):
        value = {"time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "state": state, "stage": self.stage, "identity": self.identity, **info}
        atomic_json(self.root / "status.json", value)
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def wait_downloads(self):
        self.stage = "downloads"
        started = changed = time.monotonic()
        previous = None
        while True:
            missing, progress = download_snapshot(self.plan)
            if not missing:
                break
            now = time.monotonic()
            if progress != previous:
                changed, previous = now, progress
            if now - started > self.plan["download_timeout_hours"] * 3600:
                raise Blocked("Download wait timed out; downloader/files left untouched.")
            if now - changed > self.plan["download_idle_minutes"] * 60:
                raise Blocked("Downloads made no progress within idle window; inspect download logs.")
            self.status("WAITING_DOWNLOADS", pending=missing, bytes=progress)
            time.sleep(self.plan["poll_seconds"])
        for item in self.plan["downloads"]:
            self.status("VERIFYING_DOWNLOAD", path=item["path"])
            if digest(item["path"]) != item["sha256"]:
                raise Blocked(f"Download SHA256 mismatch: {item['path']}; retained, not deleted.")

    def require_disk(self, gib):
        free = shutil.disk_usage(self.root).free / GIB
        if free < gib:
            raise Blocked(f"Disk free {free:.1f} GiB < configured threshold {gib:.1f} GiB; no files deleted.")

    def run_stage(self, name, command, outputs=(), *, budget="small", resumable=False, always=False):
        self.stage = name
        if fingerprint(self.plan) != self.identity:
            raise Blocked("Code/config/selection changed during the run; restart after review.")
        marker = self.root / (name + ".done.json")
        if marker.exists() and not always:
            saved = json.loads(marker.read_text())
            if saved["identity"] != self.identity or saved["command"] != command:
                raise Blocked(f"Completed stage identity mismatch: {name}; use a new output directory.")
            for path, checksum in saved["outputs"].items():
                if not Path(path).is_file() or digest(path) != checksum:
                    raise Blocked(f"Completed stage output changed/missing: {path}")
            self.status("SKIPPED_VERIFIED_STAGE")
            return
        if not resumable and any(Path(p).exists() for p in outputs):
            raise Blocked(f"Orphan output for {name}; inspect it or choose a new output directory.")
        self.require_disk(self.plan["disk_gib"][budget])
        self.status("RUNNING", command=command)
        started = time.monotonic()
        with (self.root / (name + ".log")).open("ab", buffering=0) as log:
            process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                while True:
                    try:
                        code = process.wait(timeout=self.plan["poll_seconds"])
                        break
                    except subprocess.TimeoutExpired:
                        self.require_disk(self.plan["disk_gib"]["reserve"])
                        elapsed = time.monotonic() - started
                        if elapsed > self.plan["stage_timeout_hours"] * 3600:
                            raise Blocked(f"Stage timeout: {name}")
                        self.status("RUNNING", child_pid=process.pid, elapsed_seconds=round(elapsed))
                if code:
                    stop_group(process)
                    raise RuntimeError(f"{name} exited {code}; inspect {name}.log")
            except BaseException:
                stop_group(process)
                raise
        checksums = {}
        for path in outputs:
            if not Path(path).is_file():
                raise RuntimeError(f"Stage exited successfully without expected output: {path}")
            checksums[str(path)] = digest(path)
        atomic_json(marker, {"identity": self.identity, "command": command, "outputs": checksums})
        self.status("STAGE_COMPLETE", duration_seconds=round(time.monotonic() - started, 1))

    def execute(self):
        smoke_only = self.plan.get("mode") == "smoke_only"
        self.wait_downloads()
        self.stage = "readiness"
        missing = prerequisites(self.plan)
        if missing:
            raise Blocked("Missing reviewed inputs: " + "; ".join(missing))
        # Check the conservative large-stage budget before loading any GPU model.
        self.require_disk(self.plan["disk_gib"]["small" if smoke_only else "calibration"])
        if not smoke_only:
            for task in self.plan["tasks"]:
                config = json.loads(Path(task["config"]).read_text())
                if config.get("provenance", {}).get("formal_calibration_approved") is False:
                    raise Blocked("Current experimental assumptions are approved for smoke only, not formal calibration.")
        py = sys.executable
        def script(name, *args):
            return [py, str(REPO / "scripts" / name), *map(str, args)]
        self.run_stage("preflight", script("overnight_preflight.py", "--plan", self.plan["plan_path"]), always=True)
        if not smoke_only:
            self.run_stage("kernels", [py, "-m", "pytest", "tests/test_w4a8.py", "tests/test_wam_w4a4.py",
                                       "tests/test_rht.py", "-ra"], always=True)
        for task in self.plan["tasks"]:
            name, config = task["name"], task["config"]
            root = self.root / name
            root.mkdir(exist_ok=True)
            obs = root / "observations.pt"
            self.run_stage(f"{name}_observations", script("prepare_calibration_data.py", "--config", config,
                           "--output", obs, "--limit", 25), [obs])
            smoke = root / "bf16_smoke.json"
            self.run_stage(f"{name}_bf16_smoke", script("overnight_preflight.py", "--config", config,
                           "--observations", obs, "--output", smoke), [smoke])
            for mode, bits, rotation in (("w4a8", 8, "none"), ("w4a4_rht", 4, "rht")):
                prefix = f"{name}_{mode}"
                folder = root / mode
                vjp = root / (mode + "_vjp_smoke")
                common = ["--config", config, "--activation-bits", bits, "--rotation", rotation,
                          "--rotation-seed", 42, "--seed", 42]
                self.run_stage(prefix + "_vjp", script("sensitivity_worker.py", *common,
                    "--observations", obs, "--output-dir", vjp, "--max-observations", 1,
                    "--num-probes", 1, "--max-rows-per-cell", 16, "--optimized", "--cache-source-dtype", "--resume"),
                    [vjp / "sensitivity.pt", vjp / "activation_cache.pt"], resumable=True)
                if smoke_only:
                    continue
                calibration = folder / "calibration.pt"
                self.run_stage(prefix + "_calibrate", script("calibrate.py", *common,
                    "--observations", obs, "--output-dir", folder, "--num-probes", 4,
                    "--max-rows-per-cell", 128, "--d-epochs", 20, "--gamma-epochs", 30,
                    "--batch-size", 128, "--recycle-every", 2),
                    [calibration, calibration.with_suffix(".json")], budget="calibration", resumable=True)
                deployment = folder / "deployment.pt"
                self.run_stage(prefix + "_export", script("export_deployment.py", "--config", config,
                    "--calibration", calibration, "--output", deployment), [deployment], budget="export")
                validation = folder / "direct_validation.json"
                self.run_stage(prefix + "_validate", script("validate_deployment.py", "--config", config,
                    "--observations", obs, "--calibration", calibration, "--deployment", deployment,
                    "--construct-device", "cpu", "--limit", 2, "--output", validation), [validation])
        self.stage = "complete"
        self.status("COMPLETE", note=("SMOKE ONLY: BF16 forward and one-observation VJP passed; no formal calibration/export/kernel validation."
                    if smoke_only else "Offline eager validation only; CUDA Graph, latency and robot SR remain separate."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Wait and execute; otherwise read-only check")
    parser.add_argument("--detach", action="store_true", help="Run supervisor in background; requires --run")
    args = parser.parse_args()
    if args.detach and not args.run:
        parser.error("--detach requires --run")
    plan = load_plan(args.plan)
    plan["plan_path"] = str(args.plan.resolve())
    if not args.run:
        pending, progress = download_snapshot(plan)
        remaining = sum(item["bytes"] - current[1] for item, current in zip(plan["downloads"], progress))
        free = shutil.disk_usage(REPO).free
        print(json.dumps({"mode": "CHECK_ONLY", "missing_approvals": prerequisites(plan),
                          "pending_downloads": pending, "download_bytes": progress,
                          "current_free_gib": round(free / GIB, 2),
                          "estimated_free_after_downloads_gib": round((free - remaining) / GIB, 2),
                          "output_dir": plan["output_dir"], "disk_thresholds_gib": plan["disk_gib"]},
                         indent=2, ensure_ascii=False))
        return 0
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root / "supervisor.log").open("ab", buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--plan", str(args.plan.resolve()), "--run"],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        print(f"Supervisor launched PID={process.pid}; check {root / 'status.json'} and supervisor.log for startup/lock errors.")
        return 0
    def interrupted(*_):
        raise KeyboardInterrupt("Supervisor termination requested")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # A common lock also prevents different plans competing for the same GPU.
    (REPO / "local").mkdir(exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock, (REPO / "local" / f"overnight_gpu_{plan['gpu']}.lock").open("a") as gpu_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = Supervisor(plan)
        (root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        try:
            supervisor.execute()
        except KeyboardInterrupt as error:
            supervisor.status("STOPPED", reason=str(error))
            return 130
        except Blocked as error:
            supervisor.status("BLOCKED", reason=str(error))
            return 2
        except Exception as error:
            supervisor.status("FAILED", reason=f"{type(error).__name__}: {error}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
