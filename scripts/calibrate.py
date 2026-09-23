#!/usr/bin/env python3
"""Single-GPU calibration, independent of robot SDK and dataset storage format."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

from fastwam_steerquant.adapters import read_config, config_identity, load_adapter
from fastwam_steerquant.adapters.observations import write_json, load_observation_records
from fastwam_steerquant.recovery import file_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--activation-bits", type=int, choices=(4, 8), required=True)
    p.add_argument("--rotation", choices=("none", "rht"), default="none")
    p.add_argument("--rotation-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-probes", type=int, default=4)
    p.add_argument("--max-rows-per-cell", type=int, default=128)
    p.add_argument("--d-epochs", type=int, default=20)
    p.add_argument("--gamma-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--recycle-every", type=int, default=2)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    if args.rotation != "none" and args.activation_bits != 4:
        p.error("RHT requires --activation-bits 4")
    if min(args.num_probes, args.max_rows_per_cell, args.batch_size) <= 0 or min(
            args.d_epochs, args.gamma_epochs, args.recycle_every) < 0:
        p.error("Invalid calibration budget")
    config = read_config(args.config)
    if config.get("provenance", {}).get("formal_calibration_approved") is False:
        raise ValueError("Experimental inputs are approved for smoke tests only; formal calibration needs review.")
    adapter = load_adapter(config)
    readiness = getattr(adapter, "require_calibration_ready", None)
    if callable(readiness):
        readiness()
    records = load_observation_records(args.observations)
    identity = {"config": config_identity(config), "observations": file_identity(args.observations),
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                             if k not in ("preflight_only", "output_dir", "config", "observations")}}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "run.json"
        if manifest.exists() and json.loads(manifest.read_text()) != identity:
            raise ValueError("Inputs/settings changed; use a different output directory.")
        print(f"Validated {len(records)} records; expected sites={config['expected_sites']}.", flush=True)
        if args.preflight_only:
            return
        write_json(manifest, identity)
        scripts = Path(__file__).resolve().parent
        common = ["--config", str(args.config.resolve()), "--activation-bits", str(args.activation_bits),
                  "--rotation", args.rotation, "--rotation-seed", str(args.rotation_seed),
                  "--seed", str(args.seed), "--resume"]
        sensitivity = [sys.executable, str(scripts / "sensitivity_worker.py"), *common,
                       "--observations", str(args.observations.resolve()), "--output-dir", str(output / "sensitivity"),
                       "--num-probes", str(args.num_probes), "--max-rows-per-cell", str(args.max_rows_per_cell),
                       "--optimized", "--cache-source-dtype", "--recycle-every", str(args.recycle_every)]
        subprocess.run([sys.executable, str(scripts / "resumable_worker.py"), "--", *sensitivity], check=True)
        subprocess.run([sys.executable, str(scripts / "calibrate_worker.py"), *common,
                        "--sensitivity", str(output / "sensitivity/sensitivity.pt"),
                        "--activation-cache", str(output / "sensitivity/activation_cache.pt"),
                        "--output", str(output / "calibration.pt"), "--rank", "0", "--world-size", "1",
                        "--d-epochs", str(args.d_epochs), "--gamma-epochs", str(args.gamma_epochs),
                        "--batch-size", str(args.batch_size)], check=True)


if __name__ == "__main__":
    main()
