#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import time
from pathlib import Path

import torch

from fastwam_steerquant import (
    ActivationCache,
    DCalibrationConfig,
    FastWAMStreamConfig,
    GammaCalibrationConfig,
    SensitivityField,
    QuantizationCheckpoint,
    calibrate_model,
    enumerate_fastwamjoint_linears,
)
from fastwam_steerquant.recovery import atomic_snapshot, file_identity, trim_memory, memory_status
from fastwam_steerquant.rht_install import install_rht_, rotation_identity
from fastwam_steerquant.adapters import read_config, load_adapter, config_identity
from fastwam_steerquant.adapters.observations import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one FastWAMJoint W4A8/W4A4 site shard.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-calls", type=int)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--d-epochs", type=int, default=20)
    parser.add_argument("--gamma-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rotation", choices=("none", "rht"), default="none")
    parser.add_argument("--rotation-seed", type=int, default=42)
    return parser.parse_args()


def balanced_site_indices(sites, world_size: int) -> list[list[int]]:
    assignments = [[] for _ in range(world_size)]
    loads = [0 for _ in range(world_size)]
    for site in sorted(sites, key=lambda item: item.module.weight.numel(), reverse=True):
        rank = min(range(world_size), key=lambda index: loads[index])
        assignments[rank].append(site.index)
        loads[rank] += site.module.weight.numel()
    return assignments


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = read_config(args.config)
    if args.num_calls is not None and args.num_calls != config["num_calls"]:
        raise ValueError("CLI num_calls differs from model configuration.")
    args.num_calls = config["num_calls"]
    if args.sigma_shift is not None and args.sigma_shift != config.get("sigma_shift"):
        raise ValueError("Set sigma_shift in the shared model configuration.")
    args.sigma_shift = config.get("sigma_shift")
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid worker rank/world size.")
    if args.rotation != "none" and args.activation_bits != 4:
        raise ValueError("RHT is supported for W4A4.")
    # Check the coordinate system before loading the expensive FP model.
    cache_header = torch.load(args.activation_cache, map_location="cpu", weights_only=True, mmap=True)
    if cache_header.get("rotation_config", {}) != rotation_identity(args.rotation, args.rotation_seed):
        raise ValueError("Activation cache rotation/seed differs from calibration; recollect rotated cache.")
    del cache_header
    field = SensitivityField.load(args.sensitivity)
    if field.rotation_config != rotation_identity(args.rotation, args.rotation_seed):
        raise ValueError("Sensitivity rotation/seed differs from calibration; recompute rotated sensitivity.")
    sensitivity_manifest = json.loads((args.sensitivity.parent / "manifest.json").read_text())
    signature = sensitivity_manifest["signature"]
    if signature["config"] != config_identity(config) or signature["activation_bits"] != args.activation_bits:
        raise ValueError("Sensitivity was collected for another model/configuration/precision.")
    if signature["num_calls"] != args.num_calls or signature["weight_bits"] != 4:
        raise ValueError("Sensitivity precision/schedule mismatch.")
    parts = args.output.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    lock = (parts / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    d_config = DCalibrationConfig(weight_bits=4, activation_bits=args.activation_bits,
                                  epochs=args.d_epochs, batch_size=args.batch_size)
    gamma_config = GammaCalibrationConfig(activation_bits=args.activation_bits,
                                          epochs=args.gamma_epochs, batch_size=args.batch_size)
    stream_config = FastWAMStreamConfig(**config["stream_config"])
    identity = {"config": config_identity(config), "sensitivity": file_identity(args.sensitivity),
                "activation_cache": file_identity(args.activation_cache), "num_calls": args.num_calls,
                "rank": args.rank, "world_size": args.world_size, "seed": args.seed,
                "d_config": asdict(d_config), "gamma_config": asdict(gamma_config)}
    identity.update(rotation_identity(args.rotation, args.rotation_seed))
    identity_path = parts / "config.json"
    had_identity = identity_path.exists()
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Calibration resume settings/artifacts changed; use a new output directory.")
    if args.resume:
        write_json(identity_path, identity)
    if args.resume and had_identity and args.output.exists():
        done = QuantizationCheckpoint.load(args.output)
        if (done.d_config != asdict(d_config) or done.gamma_config != asdict(gamma_config)
                or done.num_calls != args.num_calls
                or any(entry.rotation != args.rotation for entry in done.sites)):
            raise ValueError("Existing calibration checkpoint does not match resume settings.")
        write_json(args.output.with_suffix(".json"), {
            "rank": args.rank, "world_size": args.world_size, "sites": len(done.sites),
            "site_indices": sorted(e.site_index for e in done.sites),
            "checkpoint": str(args.output.resolve()), "recovered_final_checkpoint": True, "identity": identity})
        for entry in done.sites:
            path = parts / f"site_{entry.site_index:04d}.pt"
            if path.exists():
                path.unlink()
        print("Recovered already committed calibration checkpoint.", flush=True)
        return
    resumed = {}
    if args.resume:
        for path in sorted(parts.glob("site_*.pt")):
            shard = QuantizationCheckpoint.load(path)
            if len(shard.sites) != 1 or shard.d_config != asdict(d_config) or shard.gamma_config != asdict(gamma_config):
                raise ValueError(f"Invalid calibration site checkpoint: {path}")
            resumed[shard.sites[0].site_index] = shard.sites[0]
        print(f"Resuming {len(resumed)} completed Linear sites; {memory_status()}", flush=True)
    adapter = load_adapter(config)
    model = adapter.load_model(device="cuda").eval()
    if args.rotation != "none":
        install_rht_(model, seed=args.rotation_seed)
        print("[rotation] Calibrating rotated weights and activation caches.", flush=True)
    all_sites = enumerate_fastwamjoint_linears(model)
    if len(all_sites) != config["expected_sites"]:
        raise ValueError("Model target site count differs from configuration.")
    if field.num_calls != args.num_calls:
        raise ValueError("Sensitivity call count differs from configuration.")
    assigned = set(balanced_site_indices(all_sites, args.world_size)[args.rank])
    sites = tuple(site for site in all_sites if site.index in assigned)
    cache = ActivationCache.load(args.activation_cache)
    started = time.time()

    def progress(position: int, total: int, module_name: str) -> None:
        print(
            f"[calibrate rank={args.rank} {position + 1}/{total}] {module_name} "
            f"elapsed={(time.time() - started) / 3600:.2f}h",
            flush=True,
        )

    def commit(entry) -> None:
        if args.resume:
            QuantizationCheckpoint(4, args.activation_bits, args.num_calls, asdict(stream_config),
                                   asdict(d_config), asdict(gamma_config), (entry,)).save(
                                       parts / f"site_{entry.site_index:04d}.pt")
        trim_memory()

    checkpoint = calibrate_model(
        sites,
        cache=cache,
        sensitivity_field=field,
        stream_config=stream_config,
        d_config=d_config,
        gamma_config=gamma_config,
        progress=progress,
        resume_sites=resumed,
        on_site_complete=commit,
    )
    output = args.output.resolve()
    atomic_snapshot(checkpoint.state_dict(), output)
    write_json(
        output.with_suffix(".json"),
        {
            "rank": args.rank,
            "world_size": args.world_size,
            "sites": len(sites),
            "site_indices": sorted(assigned),
            "duration_seconds": time.time() - started,
            "identity": identity,
            "checkpoint": str(output),
        },
    )
    if args.resume:
        for entry in checkpoint.sites:
            path = parts / f"site_{entry.site_index:04d}.pt"
            if path.exists():
                path.unlink()  # Successfully consolidated into the final checkpoint.
    print(f"Calibration shard complete: {output}", flush=True)


if __name__ == "__main__":
    main()
