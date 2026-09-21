import gc

import pytest
import torch

from fastwam_steerquant.offload import BoundedSavedTensorOffload
from fastwam_steerquant.recovery import sensitivity_memory_pressure, sensitivity_commit_due


def test_memory_recycle_thresholds():
    assert sensitivity_memory_pressure(rss_limit_gib=24, cgroup_limit_gib=128,
                                       rss_bytes=13 << 30, cgroup_bytes=80 << 30) is None
    assert "worker RSS" in sensitivity_memory_pressure(rss_limit_gib=24, cgroup_limit_gib=128,
                                       rss_bytes=24 << 30, cgroup_bytes=80 << 30)
    pressure = sensitivity_memory_pressure(rss_limit_gib=24, cgroup_limit_gib=128,
                                          rss_bytes=13 << 30, cgroup_bytes=128 << 30)
    assert "cgroup memory" in pressure
    assert sensitivity_commit_due(3, 100, 1, 4, bool(pressure))
    with pytest.raises(ValueError):
        sensitivity_memory_pressure(rss_limit_gib=-1, cgroup_limit_gib=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pinned_offload_production_shape_backward_and_lifetime(dtype):
    torch.manual_seed(32)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 3072), torch.nn.GELU(),
                                torch.nn.Linear(3072, 1024)).to(device='cuda', dtype=dtype)
    model.requires_grad_(False)
    original = torch.randn(1, 294, 1024, device='cuda', dtype=dtype)
    x = original.clone().requires_grad_(True)
    model(x).float().square().mean().backward()
    expected = x.grad.clone()
    manager = BoundedSavedTensorOffload(model, max_pinned_bytes=2 << 20, min_bytes=1 << 18)
    for _ in range(8):
        x = original.clone().requires_grad_(True)
        with manager.context():
            loss = model(x).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
        del loss, x
        gc.collect()
        assert manager.active_bytes == 0
        assert manager.peak_bytes <= 2 << 20
    assert manager.pinned_copies > 0 and manager.fallback_copies > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_async_offload_stream_dependency_mutation_and_parameter_exclusion():
    model = torch.nn.Linear(64, 64, device='cuda')
    manager = BoundedSavedTensorOffload(model, max_pinned_bytes=1 << 20, min_bytes=0)
    assert manager.pack(model.weight.T).untyped_storage().data_ptr() == model.weight.untyped_storage().data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = torch.ones(512, 64, device='cuda')
        packet = manager.pack(x)
        x.add_(7)  # Enqueued AFTER the snapshot copy on the same stream.
    actual = manager.unpack(packet)  # On a different stream.
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
    del packet
    gc.collect()
    assert manager.active_bytes == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_budget_and_strided_inputs_and_exception_cleanup():
    model = torch.nn.Identity()
    manager = BoundedSavedTensorOffload(model, max_pinned_bytes=0, min_bytes=0)
    x = torch.randn(64, 32, device='cuda').T
    packet = manager.pack(x)
    torch.testing.assert_close(manager.unpack(packet), x, rtol=0, atol=0)
    assert manager.fallback_copies == 1 and manager.pinned_copies == 0
    pinned = BoundedSavedTensorOffload(model, max_pinned_bytes=1 << 20, min_bytes=0)
    packet = pinned.pack(x)
    restored = pinned.unpack(packet)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    assert restored.stride() == x.stride()
    del packet
    gc.collect()
    assert pinned.active_bytes == 0
    try:
        with pinned.context():
            y = torch.randn(64, 64, device='cuda', requires_grad=True)
            loss = y.square().sum()
            raise RuntimeError('interrupted')
    except RuntimeError:
        del loss, y
    gc.collect()
    torch.cuda.synchronize()
    assert pinned.active_bytes == 0
