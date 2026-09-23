#!/usr/bin/env python3
"""Real-model direct-vs-replacement validation, each mode in a fresh GPU process."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from fastwam_steerquant.adapters import load_adapter, read_config
from fastwam_steerquant.adapters.observations import load_observation_records, write_json
from fastwam_steerquant.checkpoint import QuantizationCheckpoint
from fastwam_steerquant.runtime import apply_checkpoint
from fastwam_steerquant.state import FastWAMCallTracker
from fastwam_steerquant.policy import QuantizedPolicy
from fastwam_steerquant.rollout_graph import LiveJointDiTGraph
from fastwam_steerquant.evaluation import assert_finite


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--deployment", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--construct-device", choices=("meta", "cpu"), default="meta")
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=("replace", "direct"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.limit < 2:
        p.error("Validate at least two changing observations")
    if args.stage is None:
        with tempfile.TemporaryDirectory(prefix="steerquant-validation-") as tmp:
            paths = []
            for stage in ("replace", "direct"):
                path = Path(tmp) / (stage + ".pt")
                command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                           "--config", args.config, "--observations", args.observations,
                           "--calibration", args.calibration, "--deployment", args.deployment,
                           "--construct-device", args.construct_device, "--limit", str(args.limit),
                           "--seed", str(args.seed), "--output", str(path)]
                if args.cuda_graph:
                    command.append("--cuda-graph")
                subprocess.run(command, check=True)
                paths.append(path)
            values = [torch.load(path, weights_only=True) for path in paths]
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
            write_json(args.output, {"direct_matches_replacement": True, "bitwise": True,
                       "observations": len(values[0]), "cuda_graph": args.cuda_graph,
                       "sr_measured": False})
        return
    config = read_config(args.config)
    adapter = load_adapter(config)
    records = load_observation_records(args.observations)[:args.limit]
    if len(records) < 2:
        raise ValueError("Provide at least two observations")
    if args.stage == "direct":
        policy = QuantizedPolicy(args.deployment, adapter, construct_device=args.construct_device,
                                 cuda_graph=args.cuda_graph)
        validator = getattr(adapter, "validate_deployment_config", None)
        if callable(validator):
            validator(policy.metadata["model_config"])
        elif policy.metadata["model_config"] != config:
            raise ValueError("Validation config differs from deployment.")
        infer = policy.infer
    else:
        model = adapter.load_model(device="cuda").eval()
        checkpoint = QuantizationCheckpoint.load(args.calibration)
        state, _ = apply_checkpoint(model, checkpoint, backend=f"cutlass_w4a{checkpoint.activation_bits}")
        del checkpoint
        if callable(getattr(adapter, "finalize_model", None)):
            adapter.finalize_model(model, device="cuda")
        tracker = FastWAMCallTracker(model, state)
        tracker.install(); model._wam_call_tracker = tracker
        if args.cuda_graph:
            LiveJointDiTGraph(model, num_calls=state.num_calls).install()
        def infer(record, *, seed):
            tracker.reset()
            kwargs = adapter.infer_kwargs(model, record, seed=seed)
            if kwargs.get("num_inference_steps") != state.num_calls:
                raise ValueError("Inference schedule differs from calibration")
            result = model.infer_action(**kwargs)
            if tracker.total_calls != state.num_calls:
                raise RuntimeError("Denoise call count mismatch")
            return result
    results = []
    with torch.inference_mode():
        for record in records:
            value = infer(record, seed=args.seed + record["sample_index"])
            assert_finite(value)
            results.append(cpu(value))
    torch.save(results, args.output)


if __name__ == "__main__":
    main()
