#!/usr/bin/env python3
"""Run each mode in a fresh process; measure observation-to-model-output latency."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from fastwam_steerquant.adapters import read_config, load_adapter
from fastwam_steerquant.adapters.observations import load_observation_records, write_json
from fastwam_steerquant.evaluation import measure_latency
from fastwam_steerquant.policy import QuantizedPolicy
from fastwam_steerquant.rollout_graph import LiveJointDiTGraph
from fastwam_steerquant.topology import enumerate_fastwamjoint_linears


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--deployment")
    p.add_argument("--output", required=True)
    p.add_argument("--cuda-graph", action="store_true")
    p.add_argument("--fuse-block", action="store_true",
                   help="Opt-in AdaLN/gate fusion; changes rounding, requires action validation.")
    p.add_argument("--actions-output", help="Optional action samples for matched-mode correctness checks")
    p.add_argument("--construct-device", choices=("meta", "cpu"), default="meta")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.limit <= 0:
        p.error("limit must be positive")
    if args.fuse_block and not args.deployment:
        p.error("--fuse-block requires a native deployment")
    for path in (args.output, args.actions_output):
        if path and Path(path).exists():
            raise FileExistsError(path)
    config = read_config(args.config)
    adapter = load_adapter(config)
    records = load_observation_records(args.observations)[:args.limit]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native deployment benchmarking requires CUDA.")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if args.deployment:
        policy = QuantizedPolicy(args.deployment, adapter, device=device,
                                 construct_device=args.construct_device, cuda_graph=args.cuda_graph,
                                 fuse_block=args.fuse_block)
        model = policy.model
        validator = getattr(adapter, "validate_deployment_config", None)
        if callable(validator):
            validator(policy.metadata["model_config"])
        elif policy.metadata["model_config"] != config:
            raise ValueError("Benchmark config differs from exported deployment config.")
        call = lambda record, seed: policy.infer(record, seed=seed)
        mode = f"w4a{policy.metadata['activation_bits']}"
    else:
        model = adapter.load_model(device=str(device)).eval()
        if any(site.module.weight.dtype != torch.bfloat16 for site in enumerate_fastwamjoint_linears(model)):
            raise ValueError("The BF16 benchmark must actually load BF16 target Linears.")
        if callable(getattr(adapter, "finalize_model", None)):
            adapter.finalize_model(model, device=device)
        if args.cuda_graph:
            LiveJointDiTGraph(model, num_calls=config["num_calls"]).install()
        def call(record, seed):
            kwargs = adapter.infer_kwargs(model, record, seed=seed)
            if kwargs.get("num_inference_steps") != config["num_calls"] or kwargs.get("sigma_shift") != config.get("sigma_shift"):
                raise ValueError("BF16 inference schedule differs from shared configuration.")
            if float(kwargs.get("text_cfg_scale", 1)) != 1:
                raise ValueError("Matched comparison requires text_cfg_scale=1.")
            for key in ("action_horizon", "num_video_frames"):
                configured = config.get("model", {}).get(key)
                if configured is not None and kwargs.get(key) != configured:
                    raise ValueError(f"BF16 inference {key} differs from configuration.")
            return model.infer_action(**kwargs)
        mode = "bf16"
    torch.cuda.synchronize(device)
    loading = {"seconds": time.perf_counter()-start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    protocol = {"config": config, "observation_ids": [r["observation_id"] for r in records],
                "observations_sha256": hashlib.sha256(open(args.observations,"rb").read()).hexdigest(),
                "cuda_graph": args.cuda_graph, "warmup": args.warmup, "repeats": args.repeats,
                "seed": args.seed, "scope": "adapter infer_kwargs + full infer_action; robot IO excluded"}
    results = []
    actions = []
    for record in records:
        stats = measure_latency(lambda: call(record, args.seed + record["sample_index"]),
                                warmup=args.warmup, repeats=args.repeats, device=device)
        results.append({"observation_id": record["observation_id"], **stats})
        if args.actions_output:
            with torch.inference_mode():
                output = call(record, args.seed + record["sample_index"])
            actions.append(output["action"].detach().float().cpu())
    graph = getattr(model, "_rollout_graph", None)
    graph_info = None
    if args.cuda_graph:
        if graph is None or len(graph.entries) != config["num_calls"] or graph.replays <= 0:
            raise RuntimeError("Requested CUDA Graph did not actually capture and replay.")
        graph_info = {"graphs": len(graph.entries), "replays": graph.replays,
                      "chunks": graph.chunks, "validation_records": graph.validation_records}
    if args.actions_output:
        Path(args.actions_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actions": torch.stack(actions), "observation_ids": protocol["observation_ids"],
                    "seeds": [args.seed + r["sample_index"] for r in records]}, args.actions_output)
    write_json(args.output, {"mode": mode, "protocol": protocol, "loading": loading,
               "runtime_settings": policy.runtime_settings if args.deployment else None,
               "optimizations": {"cuda_graph": graph_info, "fuse_block": args.fuse_block,
                                 "fused_modules": policy.fused_modules if args.deployment else 0},
               "results": results, "gpu": torch.cuda.get_device_name(device),
               "torch": str(torch.__version__), "torch_cuda": torch.version.cuda})


if __name__ == "__main__":
    main()
