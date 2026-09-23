#!/usr/bin/env python3
"""GPU/adapter gates and one-observation BF16 smoke for the overnight runner."""
import argparse
import json
import time

import torch

from fastwam_steerquant.adapters import read_config, load_adapter
from fastwam_steerquant.adapters.observations import load_observation_records, write_json
from fastwam_steerquant.evaluation import assert_finite
from fastwam_steerquant.state import DenoiseCallState, differentiable_joint_denoise
from overnight_calibration import load_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--config")
    parser.add_argument("--observations")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not accept skipped kernel tests as success.")
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This overnight plan is for the audited SM89 GPU.")
    if args.plan:
        plan = load_plan(args.plan)
        downloads = {item["path"]: item for item in plan["downloads"]}
        for task in plan["tasks"]:
            config = read_config(task["config"])
            adapter = load_adapter(config)
            adapter.require_calibration_ready()
            checkpoint = config["assets"]["source_checkpoint"]
            if downloads.get(checkpoint, {}).get("sha256") != config["real_robot"]["source_checkpoint_sha256"]:
                raise ValueError("Task checkpoint is not covered by the pinned download plan.")
            # Reuses the exact reader to check all selected file hashes, color,
            # frame bounds and evidence. No model construction here.
            records = list(adapter.dataset_records())
            if len(records) != 25:
                raise ValueError("Exactly 25 reviewed observations required per task.")
            print(f"{task['name']}: adapter and 25 observations ready", flush=True)
        return
    if not all((args.config, args.observations, args.output)):
        parser.error("Provide --plan or --config/--observations/--output")
    config = read_config(args.config)
    adapter = load_adapter(config)
    records = load_observation_records(args.observations)
    if len(records) != 25:
        raise ValueError("Expected 25 observations.")
    record = records[0]
    seed = 42 + record["sample_index"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = adapter.load_model(device="cuda").eval()
    with torch.inference_mode():
        result = model.infer_action(**adapter.infer_kwargs(model, record, seed=seed))
    assert_finite(result)
    if result["action"].shape != (32, 14):
        raise ValueError("Unexpected action shape.")
    prepared = adapter.prepare_inputs(model, record, seed=seed)
    with torch.no_grad():
        actual = differentiable_joint_denoise(model, **prepared,
            num_inference_steps=config["num_calls"], state=DenoiseCallState(config["num_calls"]),
            sigma_shift=config.get("sigma_shift"))
    torch.testing.assert_close(actual[0].float().cpu(), result["action"], rtol=0, atol=0)
    torch.cuda.synchronize()
    write_json(args.output, {"bf16_forward_finite": True, "prepared_loop_matches_infer_bitwise": True,
        "shape": [32, 14], "seed": seed, "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "vjp_test": "separate following stage"})


if __name__ == "__main__":
    main()
