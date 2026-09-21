#!/usr/bin/env python3
"""Export packed weights on CPU, from the same source/config used for calibration."""
import argparse
import json
from pathlib import Path
from fastwam_steerquant.adapters import read_config, load_adapter, config_identity
from fastwam_steerquant.checkpoint import QuantizationCheckpoint
from fastwam_steerquant.deployment import export_deployment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = read_config(args.config)
    identity = config_identity(config)
    manifest = json.loads(args.calibration.with_suffix(".json").read_text())
    if manifest.get("identity", {}).get("config") != identity:
        raise ValueError("Export config/source differs from calibration manifest.")
    checkpoint = QuantizationCheckpoint.load(args.calibration)
    if len(checkpoint.sites) != config["expected_sites"] or checkpoint.num_calls != config["num_calls"]:
        raise ValueError("Calibration topology or schedule differs from configuration.")
    adapter = load_adapter(config)
    model = adapter.load_model(device="cpu").eval()
    if any(t.device.type != "cpu" for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Export adapter must load on CPU.")
    print(export_deployment(model, checkpoint, args.output, model_config=config,
                            provenance={"calibration": str(args.calibration.resolve()), "identity": manifest["identity"]}))


if __name__ == "__main__":
    main()
