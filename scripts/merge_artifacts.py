#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from fastwam_steerquant import (
    ActivationCache,
    QuantizationCheckpoint,
    SensitivityField,
)
from fastwam_steerquant.adapters.observations import merge_observation_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge dual-GPU FastWAMJoint artifacts.")
    subparsers = parser.add_subparsers(dest="kind", required=True)
    for kind in (
        "observations",
        "sensitivity",
        "cache",
        "checkpoint",
    ):
        command = subparsers.add_parser(kind)
        command.add_argument("--inputs", type=Path, nargs="+", required=True)
        command.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) == 1 and args.kind != "observations":
        # A one-GPU run needs no tensor concatenation and no second disk copy.
        source, target = args.inputs[0].resolve(), args.output.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target:
            temporary = target.with_name(f".{target.name}.link-{os.getpid()}")
            try:
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copyfile(source, temporary)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        print(f"Single-shard {args.kind}: {target} (shared storage when on the same filesystem)")
        return
    if args.kind == "observations":
        output = merge_observation_files(args.inputs, args.output)
    elif args.kind == "sensitivity":
        output = SensitivityField.merge([SensitivityField.load(path) for path in args.inputs]).save(
            args.output
        )
    elif args.kind == "cache":
        output = ActivationCache.merge([ActivationCache.load(path) for path in args.inputs]).save(
            args.output
        )
    elif args.kind == "checkpoint":
        output = QuantizationCheckpoint.merge(
            [QuantizationCheckpoint.load(path) for path in args.inputs]
        ).save(args.output)
    print(f"Merged {args.kind}: {output}")


if __name__ == "__main__":
    main()
