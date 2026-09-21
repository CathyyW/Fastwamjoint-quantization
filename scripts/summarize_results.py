#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from fastwam_steerquant.evaluation import summarize_trials, latency_stats
from fastwam_steerquant.adapters.observations import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=("trials", "latency"))
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.kind == "trials":
        result = summarize_trials(args.inputs)
    else:
        rows = [json.loads(Path(path).read_text()) for path in args.inputs]
        fp = [r for r in rows if r["mode"] == "bf16"]
        if len(fp) != 1:
            raise ValueError("Supply exactly one matched BF16 benchmark.")
        if len({r["mode"] for r in rows}) != len(rows):
            raise ValueError("Supply one benchmark per mode.")
        result = []
        base = latency_stats([x for r in fp[0]["results"] for x in r["samples_ms"]])
        for row in rows:
            if any(row[k] != fp[0][k] for k in ("protocol", "gpu", "torch", "torch_cuda")):
                raise ValueError("Measurement protocols, GPU or software differ; speedup is not comparable.")
            stats = latency_stats([x for r in row["results"] for x in r["samples_ms"]])
            result.append({"mode": row["mode"], **stats, "speedup_mean": base["mean_ms"]/stats["mean_ms"],
                           "speedup_p50": base["p50_ms"]/stats["p50_ms"]})
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
