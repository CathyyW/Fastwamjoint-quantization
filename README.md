# FastWAM SteerQuant

FastWAMJoint action-sensitive W4A8/W4A4 calibration and native packed INT4
deployment. The W4A4 local randomized Hadamard transform is named **RHT**.

Start with **[MIGRATION_WORKFLOW.md](MIGRATION_WORKFLOW.md)** on the robot server.
It lists the implemented interfaces, outstanding robot-specific work, commands,
validation gates, and how to measure SR and speedup.

For the pack/stack real-robot **W4A8** server, use the tracked
[`real_robot_w4a8_runtime.json`](configs/real_robot_w4a8_runtime.json) profile:
tile64, CUDA Graph on, extra block fusion off. See
[server integration and no-action checks](docs/REAL_ROBOT_W4A8_RUNTIME.md).

```bash
python -m pip install -e '.[test]'
python -m pytest
python examples/cpu_smoke.py
```

Use the target FastWAM Python environment. Install a compatible PyTorch/CUDA
stack first; pyproject does not recreate FastWAM or the robot SDK environment.
The CPU example uses synthetic data and a small model. It does not evaluate the
real checkpoint or run CUDA kernels.

## Main components

- `src/fastwam_steerquant/`: retained calibration core, state/stream tracking,
  RHT, runtime replacement and live CUDA Graph.
- `kernels/w4a8/`, `kernels/w4a4/`: original fused CUDA implementations, with
  consistent RHT names in the W4A4 bindings/source.
- `deployment.py`: complete packed export and direct meta/CPU construction.
- `adapters/`: callback integration, validated observation records, reusable
  FastWAM input preparation; no LIBERO or hardware SDK imports.
- `policy.py`: serialized inference, call-state reset and optional live Graph.
- `evaluation.py`: latency statistics and explicit trial outcome recording.
- `scripts/`: calibration, resume workers, export, validation, benchmark,
  dependency setup and summaries.
- `tests/`: CPU regressions plus GPU tests that skip when CUDA is unavailable.
- `patches/`: optional existing FastWAM block-fusion integration patch.

Package name: `fastwam_steerquant`. Existing class names such as `WAMQuantLinear`
are retained for the method. Other quantization baselines, model checkpoints,
LIBERO, previous outputs and compiled binaries are not copied.

The complete deployment contains target packed INT4 weights, remaining model
state, nonpersistent registered buffers, quantization parameters and model
configuration. It does not contain external Python model code, a tokenizer's
non-tensor files or a robot SDK. A no-checkpoint model builder is required on
the target server. CPU construction is supported if meta is unavailable.

GitHub repository: [CathyyW/Fastwamjoint-quantization](https://github.com/CathyyW/Fastwamjoint-quantization).
