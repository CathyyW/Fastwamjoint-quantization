# 仿真 → 真机推理性能对齐（2026-09-24）

## 历史证据：区分正式 rollout 和额外融合 benchmark

原仓库：`/root/autodl-tmp/Fastwamjoint-quantization`。

- `docs/live_rollout_cuda_graph.md` 与 FastWAM 的 `experiments/libero/eval_libero_single.py`：正式 rollout 开启 `EVALUATION.rollout_cuda_graph=true`，使用 native kernel；没有启用额外 block fusion。
- `scripts/benchmark_fp_smoothquant_wam.py` 和 `outputs/wam_kernel_optimization/fp_smoothquant_wam_matched.json`：固定相同 BF16 捕获输入，10 步 joint DiT，总时长，不含 VAE/预处理/RPC。warmup=3，repeats=12，tile=auto。
- 该 JSON 的 `main_table_block_fusion_enabled=false`；额外 `wam_block_fused` 对照启用180个融合点：60个 ffn.0 AdaLN、60个 self_attn.o gate/residual、60个 ffn.2 gate/residual。

| 历史模式 | 10 步 DiT Graph 中位数 | 相对 BF16 Graph |
|---|---:|---:|
| BF16 | 361.684 ms | 1.000× |
| W4A8 native | 260.977 ms | 1.386× |
| W4A8 native + block fusion | 254.809 ms | 1.419× |

以上是旧仿真形状、旧 smoke 校准权重的局部性能记录，不是 stack 15k 真机延迟或 SR。旧融合检查也记录了非零数值差异，不应把额外 block fusion 当作数值无损开关。

## 迁移检查结果

当前 `kernels/w4a8/csrc/w4a8.cu`、`kernels/w4a8/ops.py`、`rollout_graph.py` 与原仓库相应文件逐字相同。不是 kernel 优化代码丢失。

之前的六帧动作误差检查使用 `cuda_graph=False`、block fusion关闭，不能拿其顺带计时代表旧 Graph 加速结果。

当前真机源快照的 MoT 不包含原仿真 `_apply_expert_post_block` 的融合分支。仅执行 `set_wam_block_fusions(model, True)` 不能使这些 kernel 自动被调用。本次增加实例级 `robot_fusion` dispatch，由 RealRobotAdapter 显式安装，不修改快照文件、模型权重或预处理。保留真机 dense core；没有盲目替换仿真 tensor-only MoT 或启用稀疏路由。

## 两个明确的部署档位

正式仿真 rollout 对齐：

```python
policy = QuantizedPolicy(
    deployment, adapter, construct_device="cpu",
    cuda_graph=True, fuse_block=False,
)
```

额外融合性能实验（W4A8）：

```python
policy = QuantizedPolicy(
    deployment, adapter, construct_device="cpu",
    cuda_graph=True, fuse_block=True,
)
assert policy.fused_modules == 180
```

融合在 Graph 安装之前完成；捕获后禁止切换。adapter必须提供经过适配的融合调用入口，否则拒绝启用。旋转 RHT 的额外 block fusion仍禁止，沿用旧限制；它自己的旋转/量化 fused producer 和 live Graph是另一回事。

第一次完整请求负责捕获并校验10张图，不纳入稳态延迟。之后使用不同观测，验证 live输入被更新，检查 `model._rollout_graph.entries` 数量和 `replays` 实际增长。不能用固定旧观测回放代替真实推理，不能忽略Graph数值检查。

`scripts/benchmark_inference.py` 新增 `--fuse-block`、`--actions-output`，输出实际Graph数、replay数、融合模块数，以及每观测预热/稳态计时。BF16与量化必须使用同一 config/obs/seed/Graph条件；额外融合的动作偏差另行报告。

## 本地验证协议

`local/run_stack_runtime_alignment.py` 串行独立进程测量6档：BF16 eager/Graph、W4A8 eager/Graph、W4A8 fused eager/Graph。每档相同2条观测、seed42/43、各3次预热和12次测量。结果位于 `outputs/stack_w4a8_runtime_alignment_v1`。本次 scope 是 adapter 输入准备 + 完整 infer_action，不含机器人/RPC；不能与旧10步DiT-only数字直接相除。

真机需复测：本服务器4090暴露约48 GiB显存，目标4090为24 GiB。Graph有额外内存池；不得把本机通过当作目标显存一定足够。保留加载、捕获和稳态的峰值记录。

本次只改本地代码，没有自动提交、推送或修改真机异步工作区。已有deployment.pt不需因启用Graph而重新校准/导出；额外融合是否采用，必须先检查动作误差和目标机运行结果。

## 本次 stack 15k 实测结果

后续DiT-only诊断发现：本节结果使用默认auto，真机M360未命中原M294优化名单，实际视频投影走tile256。显式tile64的对照及原因见 [DIT_ONLY_COMPARISON.md](DIT_ONLY_COMPARISON.md)。本节保留原始结果，不用新数据覆盖旧测量。

六档均成功。相同2条观测、每条3次预热/12次测量，合计24个稳态样本；表中是合并样本的中位数（ms）。顺序独立进程运行、GPU无其他计算任务；小规模离线诊断，不是长期真机时延认证。

| 模式 | 中位数 | p95 | 稳态最大 allocated GiB |
|---|---:|---:|---:|
| BF16 eager | 572.033 | 581.992 | 12.711 |
| BF16 Graph | 443.812 | 451.179 | 12.812 |
| W4A8 eager | 606.561 | 700.189 | 4.555 |
| W4A8 Graph | 376.754 | 379.359 | 4.651 |
| W4A8 block-fused eager | 669.818 | 732.358 | 4.555 |
| W4A8 block-fused Graph | 370.735 | 372.341 | 4.651 |

Graph对Graph：W4A8约1.178×，W4A8额外融合约1.197×。融合在Graph条件下额外降低约6.02 ms；未开启Graph的融合不能据此宣称加速。不要用572 ms的BF16 eager除以370 ms的量化Graph当作纯量化收益。

三种Graph档位各实测10张图、320次replay（含预热及额外动作留样），捕获校验严格通过。两个观测下，BF16、W4A8、W4A8融合的Graph动作各自与同模式eager逐位一致。

两个留样观测的归一化action误差：

| 对比 | RMSE | 最大绝对差 |
|---|---:|---:|
| W4A8 vs BF16 | 0.027140 | 0.111328 |
| W4A8 block-fused vs BF16 | 0.027384 | 0.101562 |
| W4A8 block-fused vs W4A8 | 0.003636 | 0.015625 |

融合不是数值无损，两个观测不替代六帧误差报告或真机SR。推荐先用Graph、不额外block融合对齐正式仿真配置，再独立评估融合是否值得采用。截图中的两次推理620 ms约束仍需目标服务器验证；本机一次370–377 ms并不证明两次能在620 ms内完成。

原始计时及Graph记录：`outputs/stack_w4a8_runtime_alignment_v1/{bf16_eager,bf16_graph,w4a8_eager,w4a8_graph,w4a8_fused_eager,w4a8_fused_graph}.json`；动作留样在对应`*_actions.pt`。这些输出及机器专用local脚本不由Git跟踪。

回归验证：启用真实源快照契约测试与CUDA后，`pytest tests -q -o addopts=''` 全部169项通过，包括历史W4A8融合数值容差/Graph测试、实例级dispatch、RHT拒绝规则、Graph安装顺序与捕获后禁止切换。
