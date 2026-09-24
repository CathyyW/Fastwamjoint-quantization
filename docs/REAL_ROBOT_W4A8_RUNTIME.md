# 真机 W4A8 运行配置：tile64 + CUDA Graph，不开额外融合

适用于pack与stack的现有W4A8 `deployment.pt`。不重新校准、不改变图像/动作预处理、不修改原BF16服务、不开启机器人动作。此配置不适用于W4A4-RHT。

## 固定设置

Git跟踪文件：`configs/real_robot_w4a8_runtime.json`。

| 参数 | 值 |
|---|---|
| W/A bits | 4 / 8 |
| rotation | none |
| construct_device | cpu |
| cuda_graph | true |
| fuse_block | false |
| w4a8_tile | 64 |

关闭的是额外AdaLN/gate-residual block fusion，不是关闭W4A8 native量化kernel或其已有producer/epilogue融合。kernel源码与二进制接口不变；本版本已有tile64环境变量，不因该配置变更而要求重新编译。模型加载时不先在GPU构建完整BF16权重。

## pull之后接入现有异步server

本机没有目标服务器新增的server/异步配置，不能声称已改了那里的YAML。量化仓库位于：

`/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant`

异步工作区位于：

`/home/agilex/World_Action_Model/physical_WM/src/fastwam-sparse/.worktrees/joint-asi-async50`

保持原server入口、RPC协议、观测/动作队列及启动工作目录不变，只在量化policy加载处使用以下方式之一。不要将本包装用于机器人执行客户端，不启动move-to-initial。

### 方式A：明确在policy构造处引用配置

```python
from pathlib import Path
from fastwam_steerquant.policy import QuantizedPolicy

quant_root = Path('/home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant')
policy = QuantizedPolicy(
    deployment_path, adapter,
    runtime_profile=quant_root / 'configs/real_robot_w4a8_runtime.json',
)
```

省略cuda_graph/fuse_block/construct_device时按profile取值。如果保留显式参数，必须分别是True/False/"cpu"，冲突会报错，而不是悄悄忽略profile。校准/其他测试未传profile且没有对应环境变量时，旧默认行为保持不变。

### 方式B：包装当前已适配的量化server命令

在原server工作目录、激活fastwam环境后，用下列前缀包装现有server命令：

```bash
bash /home/agilex/World_Action_Model/physical_WM/src/Fastwam-steerquant/bash/with_real_w4a8_runtime.sh \
  python -u <现有量化server入口> <原server参数>
```

上面尖括号是说明占位，不是可原样执行的server命令。以真机Codex确认的实际入口/配置替换，不能照搬未适配的BF16 serve_policy命令。

包装设置FASTWAM_RUNTIME_PROFILE绝对路径、FASTWAM_W4A8_TILE=64，清除EXPERIMENTAL_SMALL_TILE，保留当前工作目录及参数，并将本仓库src加入PYTHONPATH。入口必须使用新版`QuantizedPolicy`才会读取profile；若绕过它直接用load_deployment/自定义policy，需要按方式A接入，不能只设置环境变量就宣称Graph已开启。

务必停止旧server并启动新进程。C++ tile选型在首次kernel调用时缓存；已经加载过W4A8扩展、之后才启用profile会拒绝并要求重启。不要在已捕获Graph的运行实例上更改融合或tile设置。

## 无动作验收

1. 确认导入的是pull后的量化源码：

   ```bash
   python -c "import fastwam_steerquant.policy as p; print(p.__file__)"
   ```

   若安装指向其他目录，使用正确PYTHONPATH或在仓库中执行`python -m pip install -e . --no-deps`；不要为了这次配置更新重装Torch/CUDA。

2. 启动日志`[SteerQuant runtime]`应显示cuda_graph=True、fuse_block=False、w4a8_tile='64'、fused_modules=0、正确profile名称。该日志证明构造配置，尚不证明Graph实际replay。
3. 仅用离线/模拟客户端提供至少两个不同观测，不启动执行客户端。首次请求捕获10张图并严格校验；后续请求应增加replay计数：

   ```python
   assert policy.runtime_settings['w4a8_tile'] == '64'
   assert policy.runtime_settings['cuda_graph'] is True
   assert policy.runtime_settings['fuse_block'] is False
   assert policy.fused_modules == 0
   graph = policy.model._rollout_graph
   assert len(graph.entries) == policy.state.num_calls == 10
   assert graph.replays >= 20
   assert all(r['passed'] for r in graph.validation_records)
   ```

4. 比较同一观测、同一seed下原未融合W4A8与该配置的action；检查(32,14)、有限值、误差及动作反归一化仅一次。不要把与BF16的量化误差和tile/Graph实现差异混为一谈。
5. 排除第一次加载/Graph捕获再测速。BF16也开启Graph、使用相同配置；分开记录模型单次、两次合计、RPC及异步队列耗时和GPU峰值。620 ms约束仍须在目标服务器实测，不能擅自提高超时或降低执行频率来“通过”。

## 本地离线验证命令

使用机器已重定位的模型配置/观测路径（与runtime profile是不同文件）：

```bash
bash bash/with_real_w4a8_runtime.sh python scripts/benchmark_inference.py \
  --config /path/to/real_stack.json \
  --observations /path/to/observations.pt \
  --deployment checkpoint/stack_w4a8/deployment.pt \
  --construct-device cpu --cuda-graph --limit 2 --warmup 3 --repeats 12 \
  --output outputs/stack_w4a8_tile64_graph_no_fusion.json
```

不要加`--fuse-block`。pack使用对应任务的config/observations/deployment。不要让pack模型使用stack观测或统计。`benchmark_inference.py`输出runtime_settings和实际Graph检查记录，可交回校准服务器审阅。

## 本服务器验证结果（2026-09-24）

通过上面的启动包装实际加载两任务现有W4A8 deployment，每任务2个观测，seed为42/43，每观测预热3次、正式计时12次。配置日志均为tile64、Graph=True、fuse_block=False、fused_modules=0。

| 任务 | 24次完整模型调用的中位数 | 相比原未额外融合W4A8的action |
|---|---:|---|
| pack | 326.63 ms | 两个观测逐位一致，RMSE=0 |
| stack | 326.78 ms | 两个观测逐位一致，RMSE=0 |

两任务均捕获10张Graph、累计320次replay、20条严格输出校验全部通过。这里的action一致是与同任务原W4A8、相同观测/seed比较，不表示与BF16零误差，也不保证所有输入均逐位一致。关闭额外融合是因为此前stack两观测实验中，它相对未融合W4A8引入了归一化action RMSE约0.003636、最大绝对差0.015625。

原始结果位于本服务器`outputs/pack_w4a8_runtime_profile_v1/`和`outputs/stack_w4a8_runtime_profile_v1/`（benchmark.json、actions.pt；outputs不随Git发布）。pack参考`outputs/pack_action_error_6obs_v1/w4a8.pt`前两个观测；stack参考`outputs/stack_w4a8_runtime_alignment_v1/w4a8_graph_actions.pt`。

上述耗时含预处理及完整infer_action，不含RPC、异步排队或机器人执行；不是DiT-only的304.7 ms口径。不能据此断言真机两次推理满足620 ms预算，必须在目标服务器重测。

本次修改后CUDA全量测试：`python -m pytest tests -q -o addopts=''`，178 passed（37.50 s）；包括运行配置冲突检查、包装脚本、policy接入与kernel/Graph回归。
