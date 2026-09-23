import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/overnight_calibration.py"
    spec = importlib.util.spec_from_file_location("nightly_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan(tmp_path, runner):
    download = tmp_path / "model.pt"
    download.write_bytes(b"verified fixture")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"real_robot": {"task": "pack"}}))
    value = {"output_dir": str(tmp_path / "run"), "gpu": 0, "poll_seconds": 1,
            "download_timeout_hours": 1, "download_idle_minutes": 1, "stage_timeout_hours": 1,
            "disk_gib": {"small": .001, "calibration": .001, "export": .001, "reserve": .001},
            "tasks": [{"name": "pack", "config": str(config)}],
            "downloads": [{"path": str(download), "bytes": download.stat().st_size,
                           "sha256": runner.digest(download)}]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(value))
    value["plan_path"] = str(plan_path)
    return value


def test_missing_approval_is_not_invented(plan, runner):
    missing = runner.prerequisites(plan)
    assert len(missing) == 6
    assert any("normalized_action_scale" in item for item in missing)


def test_partial_is_never_complete_and_wrong_final_size_fails(plan, runner):
    path = Path(plan["downloads"][0]["path"])
    path.rename(path.with_name(path.name + ".part"))
    assert runner.download_snapshot(plan)[0] == [str(path)]
    path.write_bytes(b"short")
    with pytest.raises(runner.Blocked, match="size"):
        runner.download_snapshot(plan)


def test_bad_hash_stops_before_any_calibration(plan, runner, monkeypatch):
    plan["downloads"][0]["sha256"] = "0" * 64
    supervisor = runner.Supervisor(plan)
    calls = []
    monkeypatch.setattr(supervisor, "run_stage", lambda *args, **kw: calls.append(args))
    with pytest.raises(runner.Blocked, match="SHA256"):
        supervisor.execute()
    assert not calls


def test_missing_inputs_stop_after_download_check(plan, runner, monkeypatch):
    supervisor = runner.Supervisor(plan)
    calls = []
    monkeypatch.setattr(supervisor, "run_stage", lambda *args, **kw: calls.append(args))
    with pytest.raises(runner.Blocked, match="Missing reviewed"):
        supervisor.execute()
    assert not calls


def test_stalled_download_stops_without_deleting_partial(plan, runner, monkeypatch):
    path = Path(plan["downloads"][0]["path"])
    partial = path.with_name(path.name + ".part")
    path.rename(partial)
    supervisor = runner.Supervisor(plan)
    clock = iter([0., 0., 61.])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(runner.Blocked, match="no progress"):
        supervisor.wait_downloads()
    assert partial.exists()


def test_exit_failure_no_success_marker(plan, runner):
    supervisor = runner.Supervisor(plan)
    with pytest.raises(RuntimeError, match="exited 7"):
        supervisor.run_stage("fixture", [sys.executable, "-c", "raise SystemExit(7)"])
    assert not (supervisor.root / "fixture.done.json").exists()


def test_disk_gate_prevents_child_launch(plan, runner, monkeypatch):
    supervisor = runner.Supervisor(plan)
    monkeypatch.setattr(supervisor, "require_disk", lambda _: (_ for _ in ()).throw(runner.Blocked("disk")))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("Child must not start"))
    with pytest.raises(runner.Blocked, match="disk"):
        supervisor.run_stage("fixture", [sys.executable, "-c", "pass"])


def test_completed_output_reused_only_if_unchanged(plan, runner):
    supervisor = runner.Supervisor(plan)
    output = supervisor.root / "fixture.txt"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ok')", str(output)]
    supervisor.run_stage("fixture", command, [output])
    supervisor.run_stage("fixture", command, [output])
    assert json.loads((supervisor.root / "status.json").read_text())["state"] == "SKIPPED_VERIFIED_STAGE"
    output.write_text("changed")
    with pytest.raises(runner.Blocked, match="changed/missing"):
        supervisor.run_stage("fixture", command, [output])


def test_changed_config_blocks_resume(plan, runner):
    supervisor = runner.Supervisor(plan)
    Path(plan["tasks"][0]["config"]).write_text("{}")
    with pytest.raises(runner.Blocked, match="changed during"):
        supervisor.run_stage("fixture", [sys.executable, "-c", "pass"])


def test_serial_task_mode_order_and_fail_fast(plan, runner, monkeypatch):
    supervisor = runner.Supervisor(plan)
    monkeypatch.setattr(runner, "prerequisites", lambda _: [])
    calls = []
    monkeypatch.setattr(supervisor, "run_stage", lambda name, *a, **kw: calls.append(name))
    supervisor.execute()
    assert calls == ["preflight", "kernels", "pack_observations", "pack_bf16_smoke",
                     "pack_w4a8_vjp", "pack_w4a8_calibrate", "pack_w4a8_export", "pack_w4a8_validate",
                     "pack_w4a4_rht_vjp", "pack_w4a4_rht_calibrate", "pack_w4a4_rht_export", "pack_w4a4_rht_validate"]
    calls.clear()
    def fail(name, *a, **kw):
        calls.append(name)
        if name == "pack_w4a8_calibrate":
            raise RuntimeError("fixture failure")
    monkeypatch.setattr(supervisor, "run_stage", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        supervisor.execute()
    assert calls[-1] == "pack_w4a8_calibrate"
    assert not any(name.endswith("export") for name in calls)


def test_smoke_only_never_schedules_formal_calibration_or_export(plan, runner, monkeypatch):
    plan["mode"] = "smoke_only"
    supervisor = runner.Supervisor(plan)
    monkeypatch.setattr(runner, "prerequisites", lambda _: [])
    calls = []
    monkeypatch.setattr(supervisor, "run_stage", lambda name, *a, **kw: calls.append(name))
    supervisor.execute()
    assert calls == ["preflight", "pack_observations", "pack_bf16_smoke", "pack_w4a8_vjp", "pack_w4a4_rht_vjp"]


def test_full_plan_rejects_smoke_only_approval(plan, runner, monkeypatch):
    config = Path(plan["tasks"][0]["config"])
    config.write_text(json.dumps({"provenance": {"formal_calibration_approved": False}}))
    supervisor = runner.Supervisor(plan)
    monkeypatch.setattr(runner, "prerequisites", lambda _: [])
    monkeypatch.setattr(supervisor, "run_stage", lambda *a, **kw: pytest.fail("No stage may start"))
    with pytest.raises(runner.Blocked, match="smoke only"):
        supervisor.execute()
