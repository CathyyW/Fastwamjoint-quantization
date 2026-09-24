import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastwam_steerquant import runtime_profile as rp
from fastwam_steerquant.kernels.w4a8 import loader

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT/'configs/real_robot_w4a8_runtime.json'


@pytest.fixture
def clean_runtime(monkeypatch):
    for name in ('FASTWAM_RUNTIME_PROFILE', 'FASTWAM_W4A8_TILE', 'FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE'):
        # Record restoration even when the variable was originally absent;
        # the production helper itself writes FASTWAM_W4A8_TILE.
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)
    monkeypatch.setattr(rp, '_configured_tile', None)
    monkeypatch.setattr(loader, 'load_extension', SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=0)))
    yield monkeypatch


def test_profile_is_opt_in_and_legacy_defaults_unchanged(clean_runtime):
    options, profile = rp.resolve_runtime_profile()
    assert options == dict(construct_device='meta', cuda_graph=False, fuse_block=False)
    assert profile is None and 'FASTWAM_W4A8_TILE' not in os.environ


def test_robot_profile_sets_tile_graph_and_disables_fusion(clean_runtime):
    clean_runtime.setenv('FASTWAM_RUNTIME_PROFILE', str(PROFILE))
    options, profile = rp.resolve_runtime_profile()
    assert options == dict(construct_device='cpu', cuda_graph=True, fuse_block=False)
    assert os.environ['FASTWAM_W4A8_TILE'] == '64'
    assert profile['activation_bits'] == 8
    assert rp.resolve_runtime_profile(cuda_graph=True, fuse_block=False)[0] == options


@pytest.mark.parametrize('conflict', [dict(cuda_graph=False), dict(fuse_block=True), dict(construct_device='meta')])
def test_profile_rejects_explicit_conflicts(clean_runtime, conflict):
    with pytest.raises(ValueError, match='conflicts'):
        rp.resolve_runtime_profile(profile=PROFILE, **conflict)
    assert rp._configured_tile is None


def test_profile_rejects_environment_conflicts_and_late_load(clean_runtime):
    clean_runtime.setenv('FASTWAM_W4A8_TILE', '256')
    with pytest.raises(ValueError, match='conflicts'):
        rp.resolve_runtime_profile(profile=PROFILE)
    clean_runtime.delenv('FASTWAM_W4A8_TILE')
    clean_runtime.setenv('FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE', '0')
    with pytest.raises(ValueError, match='Unset'):
        rp.resolve_runtime_profile(profile=PROFILE)
    clean_runtime.delenv('FASTWAM_W4A8_EXPERIMENTAL_SMALL_TILE')
    clean_runtime.setattr(loader, 'load_extension', SimpleNamespace(cache_info=lambda: SimpleNamespace(currsize=1)))
    with pytest.raises(RuntimeError, match='restart'):
        rp.resolve_runtime_profile(profile=PROFILE)


def test_profile_schema_and_checkpoint_validation(clean_runtime, tmp_path):
    spec = json.loads(PROFILE.read_text())
    spec['fuse_block'] = True
    p = tmp_path/'bad.json'
    p.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='no block fusion'):
        rp.resolve_runtime_profile(profile=p)
    _, profile = rp.resolve_runtime_profile(profile=PROFILE)
    good = dict(weight_bits=4, activation_bits=8, sites=[dict(rotation='none')])
    rp.validate_profile_metadata(profile, good)
    with pytest.raises(ValueError, match='non-rotated W4A8'):
        rp.validate_profile_metadata(profile, dict(good, activation_bits=4))
    with pytest.raises(ValueError, match='non-rotated W4A8'):
        rp.validate_profile_metadata(profile, dict(good, sites=[dict(rotation='rht')]))


def test_wrapper_preserves_working_directory_and_passes_args(tmp_path):
    command = ["bash", str(ROOT/'bash/with_real_w4a8_runtime.sh'), sys.executable, '-c',
               'import json,os,sys; print(json.dumps([os.getcwd(),os.environ["FASTWAM_W4A8_TILE"],os.environ["FASTWAM_RUNTIME_PROFILE"],sys.argv[1:]]))', 'argument with spaces']
    data = json.loads(subprocess.check_output(command, cwd=tmp_path, text=True))
    assert data == [str(tmp_path), '64', str(PROFILE), ['argument with spaces']]
