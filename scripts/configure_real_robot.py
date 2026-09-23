#!/usr/bin/env python3
"""Create a local, blocked-until-reviewed config from the verified robot handoff."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from fastwam_steerquant.adapters.observations import write_json
from fastwam_steerquant.adapters.real_robot import activate_handoff, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--task", choices=("pack", "stack"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file to preserve your reviewed settings.")
    root = args.handoff.expanduser().resolve()
    manifest_hash = sha256_file(root / "SHA256SUMS")
    activate_handoff(root, manifest_hash)
    task = json.loads((root / "tasks.json").read_text())["tasks"][args.task]
    config_path = root / task["architecture_config"]
    # Resolve only architecture/processor subtrees; pretrained env paths are not used.
    yaml = OmegaConf.load(config_path)
    arch_keys = ("video_dit_config", "action_dit_config", "mot_checkpoint_mixed_attn",
                 "proprio_dim", "video_scheduler", "action_scheduler", "loss")
    arch = {key: OmegaConf.to_container(yaml.model[key], resolve=True)
            if OmegaConf.is_config(yaml.model[key]) else yaml.model[key] for key in arch_keys}
    assets = {"training_config": str(config_path),
              "dataset_stats": str(root / task["dataset_stats"]),
              "context_cache": str(root / task["context_cache_file"]),
              "handoff_manifest": str(root / "SHA256SUMS")}
    hashes = {key: sha256_file(value) for key, value in assets.items() if key != "handoff_manifest"}
    assets.update(source_checkpoint=str(args.checkpoint.expanduser().resolve()),
                  vae_checkpoint=str(args.vae.expanduser().resolve()))
    config = {
        "adapter": "fastwam_steerquant.adapters.real_robot:make_adapter",
        "num_calls": 10, "sigma_shift": None, "expected_sites": 600,
        "stream_config": {"video_latent_frames": 3, "action_chunks": 2,
                          "proprio_tokens": 1, "split_context": True},
        "model": {"dtype": "bfloat16", "action_horizon": 32, "num_video_frames": 9},
        "handoff_root": str(root), "assets": assets,
        "real_robot": {
            "task": args.task, "prompt": task["prompt"],
            "manifest_sha256": manifest_hash, "asset_sha256": hashes,
            "architecture": arch,
            "processor": OmegaConf.to_container(yaml.data.train.processor, resolve=True),
            "source_checkpoint_sha256": task["source_checkpoint"]["sha256"],
            "vae_checkpoint_sha256": sha256_file(args.vae),
        },
        "calibration_approval": {"stats_checkpoint_binding": "", "training_split": "",
                                 "image_pipeline": "", "vae_identity": ""},
        "normalized_action_scale": {"values": [], "evidence": ""},
        "dataset": {"sampling_manifest": None},
        "provenance": {"status": "ADAPTER_CONFIG_ONLY_NOT_CALIBRATION_READY",
                       "note": "Evidence fields require human-reviewed sources, not just true/false."},
    }
    write_json(args.output, config)
    print(f"Created {args.output}. Source checkpoint was NOT loaded; approval/action_scale remain pending.")


if __name__ == "__main__":
    main()
