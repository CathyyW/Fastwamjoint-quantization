import importlib.util
from pathlib import Path
import sys

import pytest


def test_subset_diagnostic_measures_clipping_without_becoming_training_scale(monkeypatch):
    pytest.importorskip("h5py")
    pytest.importorskip("PIL")
    import numpy as np
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("audit_fixture", scripts / "audit_real_robot_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = np.asarray([[-6., 2.], [6., 4.]])
    result = module.diagnose(values, {"global_mean": [0., 3.], "global_std": [1., 1.]})
    assert result["clip_fraction_per_dim"] == [1., 0.]
    np.testing.assert_allclose(result["clipped_subset_std_ddof0_DIAGNOSTIC_ONLY"], [5., 1.])
    assert "normalized_action_scale" not in result
