import pytest
import torch
import torch.nn.functional as F
from fastwam_steerquant.kernels.w4a8 import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k,n,counts", [(32,1024,4096,(16,16)), (294,3072,3072,(98,98,98)),
                                         (129,3072,14336,(128,1))])
def test_w4a8_static_schedule_and_changed_input_graph(dtype, m, k, n, counts):
    torch.manual_seed(42)
    x = torch.randn(1, m, k, device="cuda", dtype=dtype)
    q = torch.randint(-7,8,(n,k), device="cuda", dtype=torch.int8)
    packed = ops.pack_signed_int4(q)
    ws = torch.full((n,), .002, device="cuda")
    inv = torch.linspace(.5, 1.5, k, device="cuda")
    scales = torch.tensor([.02,.03], device="cuda")
    gains = torch.ones(2,len(counts), device="cuda")
    gains[1] *= 1.1
    bias = torch.randn(n, device="cuda", dtype=dtype) * .001
    def run():
        return ops.symmetric_scheduled_stream_linear(x,packed,ws,bias,inv,scales,gains,counts,1)
    with torch.inference_mode():
        actual = run()
        gain = torch.repeat_interleave(gains[1], torch.tensor(counts, device="cuda"))[None,:,None]
        qa = (x.float() * inv * (gain/scales[1])).round().clamp(-127,127)
        expected = (F.linear(qa,q.float()) * (scales[1]/gain) * ws + bias.float()).to(dtype)
        error = (actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt()
        assert error < .005
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        x.mul_(.7)
        graph.replay()
        torch.testing.assert_close(captured,run(),atol=0,rtol=0)
