#!/usr/bin/env python3
"""Run the configured dataset reader and save validated, portable tensor records."""
import argparse
from pathlib import Path
from itertools import islice
from fastwam_steerquant.adapters import load_adapter
from fastwam_steerquant.adapters.observations import save_observation_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args()
    if args.limit <= 0:
        p.error("limit must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    adapter = load_adapter(args.config)
    if not callable(getattr(adapter, "dataset_records", None)):
        raise ValueError("Connect dataset_records in the robot adapter, or use save_observation_records directly.")
    records = list(islice(adapter.dataset_records(), args.limit))
    print(save_observation_records(records, args.output))


if __name__ == "__main__":
    main()
