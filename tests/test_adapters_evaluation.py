import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from fastwam_steerquant.adapters import read_config
from fastwam_steerquant.adapters.observations import save_observation_records, load_observation_records
from fastwam_steerquant.adapters.fastwam import prepare_from_infer_kwargs
from fastwam_steerquant.evaluation import TrialRecorder, summarize_trials, latency_stats


def test_records_config_paths_and_duplicate_guard(tmp_path):
    config = {"adapter": "bridge:factory", "num_calls": 2, "expected_sites": 20,
              "stream_config": {"video_latent_frames": 1, "action_chunks": 1},
              "assets": {"source": "model.pt"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert read_config(path)["assets"]["source"] == str(tmp_path / "model.pt")
    records = [{"observation_id": "episode1_t0", "sample_index": 0, "image": torch.ones(3, 4, 4)}]
    output = save_observation_records(records, tmp_path / "obs.pt")
    torch.testing.assert_close(load_observation_records(output)[0]["image"], records[0]["image"])
    with pytest.raises(ValueError, match="Duplicate"):
        save_observation_records(records * 2, output)


def test_trial_accounting_includes_failures_and_rejects_duplicates(tmp_path):
    path = tmp_path / "trials.jsonl"
    recorder = TrialRecorder(path, mode="w4a8", protocol_id="p1", checkpoint_id="c1")
    recorder.record(task="pick", trial_id=0, success=True, reason="object placed", latencies_ms=[10,12])
    recorder.record(task="pick", trial_id=1, success=False, reason="timeout", latencies_ms=[14])
    result = summarize_trials([path])[0]
    assert result["trials"] == 2 and result["sr"] == .5
    assert result["latency"]["mean_ms"] == 12
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_trials([path,path])
    with pytest.raises(ValueError):
        latency_stats([float("nan")])


def test_prepared_noise_matches_production_independent_generators():
    model = SimpleNamespace(device="cpu", torch_dtype=torch.float32,
        vae=SimpleNamespace(temporal_downsample_factor=4, upsampling_factor=2, model=SimpleNamespace(z_dim=2)),
        action_expert=SimpleNamespace(action_dim=3),
        _encode_input_image_latents_tensor=lambda **kw: torch.ones(1,2,1,2,2),
        encode_prompt=lambda prompt: (torch.zeros(1,2,4),torch.ones(1,2,dtype=torch.bool)),
        _append_proprio_to_context=lambda **kw: (kw["context"],kw["context_mask"]))
    prepared = prepare_from_infer_kwargs(model, dict(input_image=torch.zeros(1,3,4,4),
                proprio=torch.zeros(1,3), prompt="task", num_video_frames=5, action_horizon=4), seed=19)
    expected = torch.randn((1,4,3), generator=torch.Generator().manual_seed(19))
    torch.testing.assert_close(prepared["latents_action"], expected, atol=0, rtol=0)


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_propagates_rht_and_resume_guards(tmp_path, monkeypatch):
    pipeline = script("calibrate")
    config = {"num_calls": 2, "expected_sites": 20}
    monkeypatch.setattr(pipeline, "read_config", lambda _: config)
    monkeypatch.setattr(pipeline, "config_identity", lambda _: {"fixture": 1})
    monkeypatch.setattr(pipeline, "load_adapter", lambda _: SimpleNamespace())
    observations = save_observation_records([{"observation_id":"a","sample_index":0}], tmp_path / "obs.pt")
    monkeypatch.setattr(sys, "argv", ["calibrate", "--config", str(tmp_path / "config.json"),
        "--observations", str(observations), "--output-dir", str(tmp_path / "run"),
        "--activation-bits", "4", "--rotation", "rht"])
    commands = []
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kw: commands.append(command))
    pipeline.main()
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--rotation") + 1] == "rht"
        assert "--resume" in command
    monkeypatch.setattr(pipeline, "config_identity", lambda _: {"fixture": 2})
    with pytest.raises(ValueError, match="changed"):
        pipeline.main()
