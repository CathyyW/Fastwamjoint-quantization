#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import gc
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from fastwam_steerquant.recovery import (atomic_snapshot, file_identity, memory_status,
                                              trim_memory, sensitivity_commit_due, sensitivity_memory_pressure)
from fastwam_steerquant.offload import BoundedSavedTensorOffload
from fastwam_steerquant.rht_install import install_rht_, rotation_identity

from fastwam_steerquant import (
    ActivationCache,
    ActivationCollector,
    DenoiseCallState,
    FastWAMStreamConfig,
    LazyActionSensitivityCollector,
    SensitivityAccumulator,
    differentiable_joint_denoise,
    enumerate_fastwamjoint_linears,
)
from fastwam_steerquant.adapters import read_config, load_adapter, config_identity
from fastwam_steerquant.adapters.base import validate_action_scale
from fastwam_steerquant.adapters.observations import load_observation_records, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FastWAMJoint action-sensitivity shard.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-calls", type=int)
    parser.add_argument("--num-probes", type=int, default=4)
    parser.add_argument("--max-rows-per-cell", type=int, default=256)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--max-observations", type=int, default=None,
                        help="Explicitly limit records for a sensitivity smoke run.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimized", action="store_true",
                        help="Device FP64 statistics, shared cache inputs and deduplicated input offloads.")
    parser.add_argument("--snapshot-every", type=int, default=1,
                        help="Commit every N observations; always commit before recycling and at completion.")
    parser.add_argument("--async-offload", action="store_true",
                        help="Experimental bounded pinned saved-tensor copies; no prefetch.")
    parser.add_argument("--pinned-offload-gib", type=float, default=2.)
    parser.add_argument("--recycle-rss-gib", type=float, default=0.)
    parser.add_argument("--recycle-cgroup-gib", type=float, default=0.)
    parser.add_argument("--cache-source-dtype", action="store_true")
    parser.add_argument("--recycle-every", type=int, default=0,
                        help="After N committed observations exit 75 for the supervisor to recycle this process.")
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation", choices=("none", "rht"), default="none")
    parser.add_argument("--rotation-seed", type=int, default=42)
    return parser.parse_args()


@contextmanager
def selective_saved_tensor_offload(model: torch.nn.Module, min_bytes: int = 1 << 20):
    parameter_storages = {
        parameter.untyped_storage().data_ptr()
        for parameter in model.parameters()
        if parameter.device.type == "cuda"
    }

    def pack(tensor: torch.Tensor):
        storage = tensor.untyped_storage().data_ptr()
        size_bytes = tensor.numel() * tensor.element_size()
        alignment_sensitive = tensor.ndim >= 2 and tensor.shape[-1] % 4 != 0
        if (
            tensor.device.type != "cuda"
            or storage in parameter_storages
            or size_bytes < min_bytes
            or alignment_sensitive
        ):
            return False, tensor
        return True, tensor.to(device="cpu", non_blocking=False)

    def unpack(payload):
        offloaded, tensor = payload
        if not offloaded:
            return tensor
        return tensor.to(device="cuda", non_blocking=False)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield


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
    if args.num_probes <= 0 or args.max_rows_per_cell <= 0:
        raise ValueError("Probes and cache rows must be positive.")
    if args.rotation != "none" and args.activation_bits != 4:
        raise ValueError("RHT is supported for W4A4.")
    initialization_started = time.perf_counter()
    if args.snapshot_every <= 0:
        raise ValueError("--snapshot-every must be positive.")
    if min(args.pinned_offload_gib, args.recycle_rss_gib, args.recycle_cgroup_gib) < 0:
        raise ValueError("Memory budgets must be nonnegative.")
    if (args.recycle_rss_gib or args.recycle_cgroup_gib) and not args.resume:
        raise ValueError("Memory-triggered recycling requires --resume.")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = (output_dir / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.recycle_every < 0 or (args.recycle_every and not args.resume):
        raise ValueError("Process recycling requires --resume and a nonnegative interval.")
    records = load_observation_records(args.observations)
    if args.max_observations is not None:
        if args.max_observations <= 0:
            raise ValueError("--max-observations must be positive.")
        records = records[:args.max_observations]
    signature = {
        "config": config_identity(config), "observations": file_identity(args.observations), "num_calls": args.num_calls,
        "num_probes": args.num_probes, "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits, "max_rows_per_cell": args.max_rows_per_cell,
        "cache_source_dtype": args.cache_source_dtype, "seed": args.seed,
        "sigma_shift": args.sigma_shift, "record_ids": [r["observation_id"] for r in records],
    }
    if args.optimized:
        signature["implementation"] = "device_stats_shared_capture_v2"
    signature.update(rotation_identity(args.rotation, args.rotation_seed))
    if args.async_offload:
        signature["saved_tensor_offload"] = "bounded_pinned_v1"
        signature["pinned_offload_gib"] = args.pinned_offload_gib
    snapshot_path = output_dir / "progress.pt"
    if args.resume and (output_dir / "manifest.json").exists():
        final_manifest = json.loads((output_dir / "manifest.json").read_text())
        if final_manifest.get("signature") != signature:
            raise ValueError("Existing sensitivity artifacts do not match resume settings.")
        if all((output_dir / name).is_file() for name in ("sensitivity.pt", "activation_cache.pt")):
            print("Sensitivity already complete.", flush=True)
            return
    if args.resume and (output_dir / "activation_cache.pt").exists() and not snapshot_path.exists():
        raise ValueError("Orphan final cache without matching progress/manifest; use a new output directory.")
    completed = 0
    prior_seconds = 0.
    cache = ActivationCache(num_calls=args.num_calls, max_rows_per_cell=args.max_rows_per_cell,
                            seed=args.seed, preserve_source_dtype=args.cache_source_dtype)
    cache.rotation_config = rotation_identity(args.rotation, args.rotation_seed)
    accumulator = SensitivityAccumulator(num_calls=args.num_calls)
    if args.resume and snapshot_path.exists():
        snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if snapshot["signature"] != signature:
            raise ValueError("Sensitivity resume settings/input files changed; use a new output directory.")
        completed = int(snapshot["completed"])
        prior_seconds = float(snapshot["duration_seconds"])
        cache = ActivationCache.from_state_dict(snapshot["cache"])
        accumulator = SensitivityAccumulator.from_state_dict(snapshot["accumulator"])
        del snapshot
        print(f"Resuming sensitivity at {completed}/{len(records)}; {memory_status()}", flush=True)
    adapter = load_adapter(config)
    load_started = time.perf_counter()
    print(f"[loading model] optimized={args.optimized}; {memory_status()}", flush=True)
    model = adapter.load_model(device="cuda").eval()
    if args.rotation != "none":
        install_rht_(model, seed=args.rotation_seed)
        print("[rotation] Installed official rotations before sensitivity/cache hooks.", flush=True)
    torch.cuda.synchronize()
    print(f"[initialization] model_load_seconds={time.perf_counter() - load_started:.2f} "
          f"total_seconds={time.perf_counter() - initialization_started:.2f}", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    sites = enumerate_fastwamjoint_linears(model)
    if len(sites) != config["expected_sites"]:
        raise RuntimeError(f"Expected {config['expected_sites']} quantization sites, found {len(sites)}.")
    state = DenoiseCallState(args.num_calls)
    stream_config = FastWAMStreamConfig(**config["stream_config"])
    action_std = validate_action_scale(adapter.action_scale(), device="cuda")
    started = time.time()
    committed_in_process = 0
    since_commit = 0
    pinned_offload = (BoundedSavedTensorOffload(model, max_pinned_bytes=int(args.pinned_offload_gib * 2**30))
                      if args.async_offload else None)

    with LazyActionSensitivityCollector(
        sites,
        state=state,
        stream_config=stream_config,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        accumulate_on_device=args.optimized,
        deduplicate_captures=args.optimized,
    ) as collector:
        for observation_index, record in enumerate(records):
            if observation_index < completed:
                continue
            collector.reset_observation()
            observation_started = time.perf_counter()
            timings = {"observation": observation_index + 1, "optimized": args.optimized,
                       "forward_seconds": 0., "backward_seconds": 0., "trim_seconds": 0.,
                       "snapshot_seconds": 0.}
            prepared = adapter.prepare_inputs(model, record, seed=args.seed + int(record["sample_index"]))
            required = {"latents_video", "latents_action", "first_frame_latents", "context", "context_mask"}
            if set(prepared) != required or any(not isinstance(v, torch.Tensor) for v in prepared.values()):
                raise ValueError("prepare_inputs must return the five denoising tensors.")
            if prepared["latents_action"].shape[-1] != action_std.numel():
                raise ValueError("action_scale dimension differs from model action dimension.")
            torch.cuda.synchronize()
            timings["prepare_seconds"] = time.perf_counter() - observation_started
            probe_generator = torch.Generator(device="cuda").manual_seed(
                args.seed * 1000003 + int(record["sample_index"])
            )
            for probe_index in range(args.num_probes):
                collector.begin_probe()
                run_inputs = {
                    key: value.detach().clone()
                    for key, value in prepared.items()
                }
                for key in ("latents_video", "latents_action", "first_frame_latents", "context"):
                    run_inputs[key].requires_grad_(True)
                activation_context = (
                    ActivationCollector(
                        sites,
                        state=state,
                        stream_config=stream_config,
                        cache=cache,
                    )
                    if probe_index == 0 and not args.optimized
                    else None
                )
                if activation_context is not None:
                    activation_context.install()
                collector.activation_cache = cache if args.optimized and probe_index == 0 else None
                phase_started = time.perf_counter()
                try:
                    offload_context = (pinned_offload.context() if pinned_offload is not None
                                       else selective_saved_tensor_offload(model))
                    with offload_context:
                        final_action = differentiable_joint_denoise(
                            model,
                            **run_inputs,
                            num_inference_steps=args.num_calls,
                            state=state,
                            sigma_shift=args.sigma_shift,
                        )
                finally:
                    if activation_context is not None:
                        activation_context.remove()
                    collector.activation_cache = None
                torch.cuda.synchronize()
                timings["forward_seconds"] += time.perf_counter() - phase_started
                collector.validate_forward()
                signs = (
                    torch.empty_like(final_action, dtype=torch.float32)
                    .bernoulli_(0.5, generator=probe_generator)
                    .mul_(2)
                    .sub_(1)
                )
                scalar = ((final_action.float() / action_std) * signs).sum()
                scalar = scalar / math.sqrt(final_action.numel())
                phase_started = time.perf_counter()
                scalar.backward()
                torch.cuda.synchronize()
                timings["backward_seconds"] += time.perf_counter() - phase_started
                del scalar, signs, final_action, run_inputs
                phase_started = time.perf_counter()
                trim_memory()
                timings["trim_seconds"] += time.perf_counter() - phase_started
                print(f"[probe {observation_index + 1}/{len(records)} {probe_index + 1}/{args.num_probes}] "
                      f"{memory_status()}", flush=True)

            observed = collector.finish_observation(args.num_probes)
            for site in sites:
                names, values = observed[site.index]
                accumulator.add(site, names, values)
            del prepared, observed
            trim_memory()
            elapsed = time.time() - started
            print(
                f"[sensitivity {observation_index + 1}/{len(records)}] "
                f"{record['observation_id']} elapsed={elapsed / 3600:.2f}h "
                f"peak={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB {memory_status()}",
                flush=True,
            )
            completed = observation_index + 1
            committed_in_process += 1
            since_commit += 1
            recycle_due = bool(args.recycle_every and committed_in_process >= args.recycle_every
                               and completed < len(records))
            pressure = sensitivity_memory_pressure(rss_limit_gib=args.recycle_rss_gib,
                                                   cgroup_limit_gib=args.recycle_cgroup_gib)
            if pressure and completed < len(records):
                recycle_due = True
                print(f"[memory recycle] {pressure}; committing before exit", flush=True)
            if args.resume and sensitivity_commit_due(completed, len(records), since_commit,
                                                       args.snapshot_every, recycle_due):
                phase_started = time.perf_counter()
                atomic_snapshot({"signature": signature, "completed": completed,
                                 "duration_seconds": prior_seconds + time.time() - started,
                                 "cache": cache.state_dict(), "accumulator": accumulator.state_dict()},
                                snapshot_path)
                timings["snapshot_seconds"] = time.perf_counter() - phase_started
                since_commit = 0
                print(f"[committed] {completed}/{len(records)}", flush=True)
            timings["total_seconds"] = time.perf_counter() - observation_started
            if pinned_offload is not None:
                timings["pinned_active_bytes"] = pinned_offload.active_bytes
                timings["pinned_peak_bytes"] = pinned_offload.peak_bytes
                timings["pinned_copies"] = pinned_offload.pinned_copies
                timings["pageable_fallback_copies"] = pinned_offload.fallback_copies
            print("[timing] " + json.dumps(timings, sort_keys=True), flush=True)
            with (output_dir / "timings.jsonl").open("a") as timing_file:
                timing_file.write(json.dumps(timings, sort_keys=True) + "\n")
            if recycle_due:
                print("[recycle] Progress saved; requesting a fresh worker (exit 75).", flush=True)
                raise SystemExit(75)

    field_path = accumulator.finalize(rotation_config=cache.rotation_config).save(output_dir / "sensitivity.pt")
    cache_path = output_dir / "activation_cache.pt"
    # This file is only produced after all observations are committed. A
    # successful atomic final save can be reused if the worker died afterward.
    if not (args.resume and cache_path.exists()):
        atomic_snapshot(cache.state_dict(), cache_path)
    write_json(
        output_dir / "manifest.json",
        {
            "observations": len(records),
            "num_probes": args.num_probes,
            "num_calls": args.num_calls,
            "sites": len(sites),
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "max_rows_per_cell": args.max_rows_per_cell,
            "sensitivity": str(field_path),
            "activation_cache": str(cache_path),
            "duration_seconds": prior_seconds + time.time() - started,
            "signature": signature, "preserve_source_dtype": args.cache_source_dtype,
            "runtime": {"recycle_every": args.recycle_every, "snapshot_every": args.snapshot_every,
                        "recycle_rss_gib": args.recycle_rss_gib,
                        "recycle_cgroup_gib": args.recycle_cgroup_gib,
                        "async_offload": args.async_offload, "pinned_offload_gib": args.pinned_offload_gib},
        },
    )
    if args.resume and snapshot_path.exists():
        snapshot_path.unlink()  # Temporary progress is now represented by the final artifacts.
    print(f"Sensitivity shard complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
