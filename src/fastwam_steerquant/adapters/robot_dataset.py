"""Read reviewed raw-HDF calibration frames; no implicit split/color guesses."""
import json
from pathlib import Path

import torch

from .real_robot import verify_file


def records_from_manifest(config):
    import h5py
    import numpy as np

    path = config.get("dataset", {}).get("sampling_manifest")
    if not path:
        raise ValueError("Provide a reviewed dataset.sampling_manifest before extracting observations.")
    path = Path(path).expanduser().resolve()
    manifest = json.loads(path.read_text())
    task = config["real_robot"]
    if manifest.get("task") != task["task"]:
        raise ValueError("Sampling manifest task mismatch.")
    for key in ("training_split_evidence", "image_pipeline_evidence"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Sampling requires {key}.")
    color = manifest.get("source_color_order")
    if color not in ("RGB", "BGR", "as_stored"):
        raise ValueError("Verify source_color_order RGB/BGR or explicitly approve as_stored.")
    if color == "as_stored" and config.get("dataset", {}).get("channel_policy") != "as_stored":
        raise ValueError("as_stored requires explicit dataset.channel_policy approval.")
    episodes = manifest["episodes"]
    if len(episodes) != 3 or [len(e["frames"]) for e in episodes] != [9, 8, 8]:
        raise ValueError("This experiment requires 3 episodes with 9+8+8 explicit frame selections.")
    seen_paths, seen_hashes = set(), set()
    resolved = []
    # Complete integrity/selection checks before yielding any record.
    for episode in episodes:
        source = (path.parent / episode["path"]).resolve()
        if source.suffix != ".hdf5" or source in seen_paths or episode["sha256"] in seen_hashes:
            raise ValueError("Use three distinct completed .hdf5 files, never .part files.")
        seen_paths.add(source)
        seen_hashes.add(episode["sha256"])
        verify_file(source, episode["sha256"])
        frames = episode["frames"]
        indices = [frame["index"] for frame in frames]
        if any(type(i) is not int or i < 0 for i in indices) or indices != sorted(set(indices)):
            raise ValueError("Frame indices must be unique, sorted, nonnegative integers.")
        if any(not isinstance(f.get("stage"), str) or not f["stage"].strip() for f in frames):
            raise ValueError("Each frame needs a reviewed task-stage label.")
        with h5py.File(source, "r") as file:
            states = file["observations/qpos"]
            if states.ndim != 2 or states.shape[1] != 14 or indices[-1] >= len(states):
                raise ValueError("Invalid HDF state shape or selected frame index.")
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                images = file[f"observations/images/{name}"]
                if images.shape != (len(states), 480, 640, 3) or images.dtype != np.uint8:
                    raise ValueError("Only synchronized, uncompressed uint8 HWC camera datasets are supported.")
        resolved.append((source, episode))
    sample = 0
    for source, episode in resolved:
        with h5py.File(source, "r") as file:
            for frame in episode["frames"]:
                index = frame["index"]
                images = {}
                for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                    value = file[f"observations/images/{name}"][index]
                    if color == "BGR":
                        value = value[..., ::-1]
                    images[name] = torch.from_numpy(value.copy())
                state = torch.from_numpy(file["observations/qpos"][index].astype(np.float32))
                if not torch.isfinite(state).all():
                    raise ValueError("Nonfinite selected state.")
                yield {"sample_index": sample,
                       "observation_id": f"{task['task']}:{episode['sha256']}:{index}",
                       "task": task["task"], "prompt": task["prompt"],
                       "image_channel_order": "as_stored" if color == "as_stored" else "RGB",
                       "images": images, "state": state,
                       "provenance": {"source": str(source), "sha256": episode["sha256"],
                                      "frame_index": index, "stage": frame["stage"],
                                      "source_color_order": color,
                                      "training_split_evidence": manifest["training_split_evidence"],
                                      "image_pipeline_evidence": manifest["image_pipeline_evidence"]}}
                sample += 1
