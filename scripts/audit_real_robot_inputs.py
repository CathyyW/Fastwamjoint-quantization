#!/usr/bin/env python3
"""Local evidence audit only: no GPU, calibration, inferred approvals or downloads."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from fastwam_steerquant.adapters import read_config, load_adapter
from fastwam_steerquant.adapters.observations import write_json
from fastwam_steerquant.adapters.real_robot import sha256_file, verify_file
from overnight_calibration import load_plan

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def primitive(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def diagnose(values, stats):
    mean, std = (np.asarray(stats[f"global_{key}"], dtype=np.float64) for key in ("mean", "std"))
    z = values * (1 / (std + 1e-8)) - mean / (std + 1e-8)
    return {"raw_min": values.min(0).tolist(), "raw_max": values.max(0).tolist(),
            "clip_fraction_per_dim": (np.abs(z) > 5).mean(0).tolist(),
            "clipped_subset_std_ddof0_DIAGNOSTIC_ONLY": np.clip(z, -5, 5).std(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strict-model", action="store_true", help="Strict CPU loading of each task checkpoint")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = load_plan(args.plan)
    downloads = {item["path"]: item for item in plan["downloads"]}
    report = {"kind": "local_input_evidence_not_calibration_approval", "tasks": {}, "vae": {}}
    model = None
    model_architecture = None
    for task in plan["tasks"]:
        name = task["name"]
        config = read_config(task["config"])
        adapter = load_adapter(config)
        task_dir = output / name
        task_dir.mkdir()
        checkpoint = config["assets"]["source_checkpoint"]
        print(f"{name}: hashing downloaded task checkpoint", flush=True)
        verify_file(checkpoint, config["real_robot"]["source_checkpoint_sha256"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if set(payload) != {"mot", "proprio_encoder", "step", "torch_dtype"}:
            raise ValueError("Unexpected checkpoint schema")
        assert payload["step"] == 15000
        assert len(payload["mot"]) == 1649
        assert all(t.dtype == torch.bfloat16 for section in ("mot", "proprio_encoder") for t in payload[section].values())
        result = {"checkpoint_sha256_matches_robot": True,
                  "checkpoint_sha256": config["real_robot"]["source_checkpoint_sha256"],
                  "checkpoint_bytes": Path(checkpoint).stat().st_size,
                  "checkpoint_keys": sorted(payload), "step": payload["step"],
                  "mot_tensors": len(payload["mot"]), "mot_elements": sum(t.numel() for t in payload["mot"].values()),
                  "all_weights_bf16": True, "contains_training_metadata": False,
                  "stats_sha256": sha256_file(config["assets"]["dataset_stats"]),
                  "context_shape": list(adapter.context.shape),
                  "context_all_true_mask": bool(adapter.context_mask.all()), "episodes": []}
        if args.strict_model:
            print(f"{name}: strict CPU model loading (no forward)", flush=True)
            if model is None:
                model = adapter.build_model(config, device="cpu")
                model_architecture = config["real_robot"]["architecture"]
            elif model_architecture != config["real_robot"]["architecture"]:
                raise ValueError("Tasks have different architectures; audit them separately.")
            model.mot.load_state_dict(payload["mot"], strict=True)
            model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            result["strict_cpu_mot_proprio_load"] = True
        del payload
        vae_path = config["assets"]["vae_checkpoint"]
        if vae_path not in report["vae"]:
            verify_file(vae_path, config["real_robot"]["vae_checkpoint_sha256"])
            report["vae"][vae_path] = {"sha256": config["real_robot"]["vae_checkpoint_sha256"],
                "bytes": Path(vae_path).stat().st_size, "robot_hash_provided": False}
        stats = json.loads(Path(config["assets"]["dataset_stats"]).read_text())
        a = stats["action"]["default"]
        mean, std = (np.asarray(a[f"global_{key}"], dtype=np.float64) for key in ("mean", "std"))
        extremum = np.maximum(np.abs((np.asarray(a["global_min"]) - mean) / (std + 1e-8)),
                              np.abs((np.asarray(a["global_max"]) - mean) / (std + 1e-8)))
        result["stats_action_max_abs_z"] = extremum.tolist()
        result["stats_action_dims_potentially_clipped"] = np.flatnonzero(extremum > 5).tolist()
        result["unclipped_action_std_CONDITIONAL_on_training_stats"] = (std / (std + 1e-8)).tolist()
        dataset_name = "pack_3_objects_plus" if name == "pack" else "stack_3_cups_gen"
        files = [item for path, item in downloads.items() if f"/{dataset_name}/" in path]
        files.sort(key=lambda item: int(Path(item["path"]).stem.split("_")[-1]))
        assert len(files) == 3
        draft = {"task": name, "status": "DRAFT_NOT_APPROVED", "source_color_order": None,
                 "training_split_evidence": "", "image_pipeline_evidence": "", "episodes": []}
        all_action, all_state = [], []
        for item, count in zip(files, (9, 8, 8)):
            path = Path(item["path"])
            print(f"{name}: verifying and inspecting {path.name}", flush=True)
            if path.stat().st_size != item["bytes"]:
                raise ValueError("HDF size mismatch")
            verify_file(path, item["sha256"])
            with h5py.File(path, "r") as file:
                state = np.asarray(file["observations/qpos"], dtype=np.float64)
                action = np.asarray(file["action"], dtype=np.float64)
                assert state.shape == action.shape and state.shape[1] == 14
                assert np.isfinite(state).all() and np.isfinite(action).all()
                all_state.append(state)
                all_action.append(action)
                n = len(state)
                # Candidate time-stratified selection, not semantic-stage approval.
                indices = np.rint(np.linspace(0, n - 1, count)).astype(int).tolist()
                episode = {"path": str(path), "sha256": item["sha256"], "frames": [
                    {"index": index, "stage": f"time_bin_{i+1}_of_{count}; semantic stage pending review"}
                    for i, index in enumerate(indices)]}
                draft["episodes"].append(episode)
                sheet = Image.new("RGB", (count * 192, 3 * 166), "white")
                draw = ImageDraw.Draw(sheet)
                schema = {}
                for row, camera in enumerate(CAMERAS):
                    dataset = file[f"observations/images/{camera}"]
                    assert dataset.shape == (n, 480, 640, 3) and dataset.dtype == np.uint8
                    schema[camera] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype),
                                      "attrs": {k: primitive(v) for k, v in dataset.attrs.items()}}
                    for col, index in enumerate(indices):
                        # Data-inspection visualization of stored channels. No color-order assumption is approved.
                        thumbnail = Image.fromarray(dataset[index]).resize((192, 144))
                        sheet.paste(thumbnail, (col * 192, row * 166 + 22))
                        draw.text((col * 192 + 2, row * 166 + 3), f"{camera} t={index}", fill="black")
                preview = task_dir / (path.stem + "_stored_channels.jpg")
                sheet.save(preview, quality=92)
                result["episodes"].append({"path": str(path), "sha256_verified": True, "frames": n,
                    "root_attrs": {k: primitive(v) for k, v in file.attrs.items()},
                    "root_keys": list(file.keys()), "cameras": schema, "selected_candidates": indices,
                    "preview": str(preview), "state_diagnostic": diagnose(state, stats["state"]["default"]),
                    "action_diagnostic": diagnose(action, a)})
        result["three_episode_action_diagnostic_NOT_training_statistics"] = diagnose(np.concatenate(all_action), a)
        result["three_episode_state_diagnostic"] = diagnose(np.concatenate(all_state), stats["state"]["default"])
        write_json(task_dir / "sampling.draft.json", draft)
        report["tasks"][name] = result
        write_json(output / "audit.json", report)
    report["cuda_initialized"] = torch.cuda.is_initialized()
    write_json(output / "audit.json", report)
    print(f"Audit complete: {output / 'audit.json'}; no approvals/configs were changed", flush=True)


if __name__ == "__main__":
    main()
