# Stack 15k W4A8：DiT-only 与 tile 选型诊断

2026-09-24，本服务器离线测试，无机器人/RPC操作。checkpoint与前次完整模型调用基准相同。

## 结论

之前的约1.20×不能主要解释为增加VAE等计时范围：相同默认选型下，单测DiT也仅约1.21×。进一步检查发现，kernel自动选型仅覆盖旧仿真 `M=294` 视频token形状，真机 `M=360` 未匹配优化名单，走tile256。用已有的 `FASTWAM_W4A8_TILE=64` 显式覆盖后，DiT-only恢复约1.42×。

没有修改生产kernel默认选择，也没有重新校准、导出或上传checkpoint。tile是GEMM计算分块，不是量化bit数或校准参数。

## 方法和范围

直接复用旧仓库 `baseline/dit_replay.py` 的 `JointDiTReplay.run_graph`，原文件SHA256记录在JSON中。每轮从BF16完整推理捕获两个观测各10步core输入；BF16、W4A8、W4A8 block-fused重复使用相同输入。每观测3轮预热、12轮测量，合计24个10步序列，用CUDA events计时，汇总中位数。

计时包含10步video/action joint DiT core；不含VAE、输入预处理、prompt编码、scheduler更新、live Graph输入拷贝、动作CPU回传或RPC。是固定输入性能benchmark，不是完整动作生成延迟或成功率测试。

## 实测

| 选型 | BF16 DiT | W4A8 DiT | W4A8加速 | W4A8融合DiT | 融合加速 |
|---|---:|---:|---:|---:|---:|
| 默认auto（视频M360走256） | 421.110 ms | 354.918 ms | 1.186× | 349.284 ms | 1.206× |
| 显式tile64 | 421.322 ms | 304.672 ms | 1.383× | 297.197 ms | 1.418× |

两轮输入ID、seed、预热/重复次数、core输入形状、GPU、Torch版本及历史计时器SHA256一致。BF16基线约0.05%的波动，量化的变化明显大于基线波动。

对比历史仿真extra block-fused benchmark：361.684 / 254.809 = 1.419×。旧输入224×448对应视频294 tokens，当前384×320对应视频360 tokens；action32 tokens、context129 tokens。绝对耗时不能直接套用，但优化分支对齐后本次测得的倍数接近。

## 代码证据与部署注意

`kernels/w4a8/csrc/w4a8.cu` 的 `launch_packed_w4a8_gemm`：

- `auto`只对列出的M32/M129/M294生产矩阵形状，在SM89上使用tile64。
- 未匹配者使用tile256，包括当前视频M360。
- `FASTWAM_W4A8_TILE=64`支持进程级显式覆盖；开关在首次调用时缓存，必须在server启动前设置并重启进程，不能在已捕获Graph上动态切换。
- 后续要改变auto覆盖名单，需专门校验形状性能与数值；本次仅做显式环境变量对照，没有改名单。

## 产物

- `local/benchmark_stack_dit_alignment.py`：本机诊断入口，依赖旧仓库计时器路径。
- `outputs/stack_w4a8_dit_alignment_v1/results.json`：auto。
- `outputs/stack_w4a8_dit_tile64_alignment_v1/results.json`：tile64。

这些机器专用local脚本与outputs默认不进Git；本报告可随代码交接。完整模型口径见 `PERFORMANCE_ALIGNMENT.md`，不要把DiT-only数字标成真机端到端。

## 补充：完整模型调用与动作核验

同样2观测、seed42/43、每观测3次预热+12次测量，另开进程设置 `FASTWAM_W4A8_TILE=64`，运行 `benchmark_inference.py --cuda-graph --fuse-block`：

- 完整调用中位数319.035 ms，均值321.234 ms，p95 337.793 ms。
- 对照上一轮相同协议BF16 Graph中位数443.812 ms，约1.391×（顺序独立进程测量，不是同时计时）。
- 180个额外融合点，10张Graph、320次replay，20项捕获数值检查严格通过。
- 两个观测下，tile64的最终action与auto选型、相同融合/Graph模式的action逐位一致，RMSE和max abs均为0。这里只证明这两个样本的tile切换一致性；额外block fusion与未融合之间仍存在前报告所述舍入差异。

产物：`outputs/stack_w4a8_dit_tile64_alignment_v1/full_fused_graph.json`、`full_fused_graph_actions.pt`。本次没有改变生产默认auto名单或真机配置。319 ms的模型调用不含RPC/机器人IO，不能据此认证真机两次推理620 ms时限。
