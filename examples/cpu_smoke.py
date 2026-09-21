"""CPU integration example: actual sensitivity/calibration -> export -> packed load.

Synthetic topology/data only. It validates plumbing, NOT real-model quality/SR.
Run from the repository root: python examples/cpu_smoke.py
"""
from pathlib import Path
import torch
from torch import nn
from fastwam_steerquant import (
    ActivationCache, ActivationCollector, FastWAMStreamConfig, DenoiseCallState,
    SensitivityAccumulator, DCalibrationConfig, GammaCalibrationConfig,
    enumerate_fastwamjoint_linears, estimate_action_sensitivity, calibrate_model,
)
from fastwam_steerquant.deployment import export_deployment, load_deployment
from fastwam_steerquant.topology import resolve_site_module
from fastwam_steerquant.runtime import apply_checkpoint


class Attention(nn.Module):
    def __init__(self, width):
        super().__init__()
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(width, width, dtype=torch.bfloat16))


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(width), Attention(width)
        self.ffn = nn.Sequential(nn.Linear(width, width, dtype=torch.bfloat16), nn.GELU(),
                                 nn.Linear(width, width, dtype=torch.bfloat16))


def build_model(config, *, device):
    with torch.device(device):
        model = nn.Module()
        for name in ("video_expert", "action_expert"):
            expert = nn.Module()
            expert.blocks = nn.ModuleList([Block(config["width"])])
            setattr(model, name, expert)
    return model


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model({"width": 32}, device="cpu")
    sites = enumerate_fastwamjoint_linears(model)
    state = DenoiseCallState(1)
    streams = FastWAMStreamConfig(video_latent_frames=1, action_chunks=1, split_context=False)
    x = torch.randn(1, 6, 32, dtype=torch.bfloat16, requires_grad=True)
    def trajectory():
        state.set(0)
        return sum(site.module(x).float() for site in sites)
    cache = ActivationCache(num_calls=1, max_rows_per_cell=16)
    with ActivationCollector(sites, state=state, stream_config=streams, cache=cache):
        observed = estimate_action_sensitivity(trajectory, sites, state=state, stream_config=streams,
                    action_scale=torch.ones(32), weight_bits=4, activation_bits=8, num_probes=1)
    accumulator = SensitivityAccumulator(num_calls=1)
    for site in sites:
        accumulator.add(site, *observed[site.index])
    checkpoint = calibrate_model(sites, cache=cache, sensitivity_field=accumulator.finalize(),
                stream_config=streams, d_config=DCalibrationConfig(epochs=1, batch_size=6),
                gamma_config=GammaCalibrationConfig(epochs=1, batch_size=6))
    output = Path("outputs/cpu_smoke/deployment.pt")
    if output.exists():
        raise FileExistsError("Choose a fresh outputs/cpu_smoke directory before repeating the example.")
    export_deployment(model, checkpoint, output, model_config={"width": 32})
    packed, _, _ = load_deployment(output, build_model, device="cpu")
    apply_checkpoint(model, checkpoint, backend="cutlass_w4a8")
    for entry in checkpoint.sites:
        a = resolve_site_module(packed, entry.module_name)
        b = resolve_site_module(model, entry.module_name)
        for name, value in b.state_dict().items():
            torch.testing.assert_close(a.state_dict()[name], value, atol=0, rtol=0)
        assert a.weight.dtype == torch.uint8
    print(f"PASS: sensitivity -> calibration -> packed export -> meta load ({len(sites)} sites).")
    print(f"Deployment: {output}; CPU smoke does not execute CUDA kernels or measure SR.")


if __name__ == "__main__":
    main()
