import json

import pytest
import torch

from fastwam_steerquant.adapters.real_robot import sha256_file
from fastwam_steerquant.adapters.robot_dataset import records_from_manifest


@pytest.fixture
def selected(tmp_path):
    h5py = pytest.importorskip("h5py")
    episodes = []
    for i, count in enumerate((9, 8, 8)):
        path = tmp_path / f"episode_{i}.hdf5"
        with h5py.File(path, "w") as file:
            file.attrs["fixture"] = i
            file.create_dataset("observations/qpos", shape=(12, 14), dtype="f4", fillvalue=i)
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                images = file.create_dataset(f"observations/images/{camera}",
                    shape=(12, 480, 640, 3), dtype="u1", fillvalue=0)
                if i == 0:
                    images[0, :, :, 0] = 255
        episodes.append({"path": path.name, "sha256": sha256_file(path),
                         "frames": [{"index": f, "stage": "fixture"} for f in range(count)]})
    manifest = {"task": "pack", "source_color_order": "BGR", "episodes": episodes,
                "training_split_evidence": "synthetic fixture", "image_pipeline_evidence": "synthetic fixture"}
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(manifest))
    config = {"real_robot": {"task": "pack", "prompt": "test"}, "dataset": {"sampling_manifest": str(path)}}
    return config, manifest, path


def test_25_reviewed_records_have_unique_ids_and_rgb_conversion(selected):
    config, _, _ = selected
    records = list(records_from_manifest(config))
    assert len(records) == len({r["observation_id"] for r in records}) == 25
    assert [r["sample_index"] for r in records] == list(range(25))
    image = records[0]["images"]["cam_high"]
    assert image.dtype == torch.uint8
    assert (image[..., 2] == 255).all() and (image[..., :2] == 0).all()
    assert [r["state"][0].item() for r in records] == [0.] * 9 + [1.] * 8 + [2.] * 8


def test_as_stored_requires_approval_and_never_swaps_channels(selected):
    config, manifest, path = selected
    manifest["source_color_order"] = "as_stored"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="channel_policy"):
        next(records_from_manifest(config))
    config["dataset"]["channel_policy"] = "as_stored"
    record = next(records_from_manifest(config))
    assert record["image_channel_order"] == "as_stored"
    assert (record["images"]["cam_high"][..., 0] == 255).all()
    assert (record["images"]["cam_high"][..., 2] == 0).all()


def test_explicit_counts_override_default_and_exclusion_is_preflight(selected):
    config, manifest, path = selected
    config["dataset"]["per_episode_counts"] = [3, 2, 2]
    for ep, count in zip(manifest["episodes"], [3, 2, 2]):
        ep["frames"] = ep["frames"][:count]
    path.write_text(json.dumps(manifest))
    assert len(list(records_from_manifest(config))) == 7
    last = manifest["episodes"][-1]
    manifest["excluded_observation_ids"] = [f"pack:{last['sha256']}:1"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="excluded"):
        next(records_from_manifest(config))


@pytest.mark.parametrize("counts", [[25,25], [25,25,0], [True,25,25], "75"])
def test_invalid_explicit_counts(selected, counts):
    config, _, _ = selected
    config["dataset"]["per_episode_counts"] = counts
    with pytest.raises(ValueError, match="per_episode_counts"):
        next(records_from_manifest(config))


@pytest.mark.parametrize("damage", ["split", "color", "hash", "frame", "duplicate"])
def test_invalid_sampling_manifest_rejected_before_first_record(selected, damage):
    config, manifest, path = selected
    if damage == "split":
        manifest["training_split_evidence"] = ""
    elif damage == "color":
        manifest["source_color_order"] = None
    elif damage == "hash":
        manifest["episodes"][2]["sha256"] = "0" * 64
    elif damage == "frame":
        manifest["episodes"][2]["frames"][-1]["index"] = 99
    else:
        manifest["episodes"][2].update(path=manifest["episodes"][1]["path"], sha256=manifest["episodes"][1]["sha256"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        next(records_from_manifest(config))
