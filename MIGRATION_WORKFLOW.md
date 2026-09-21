# FastWAM SteerQuant：真机服务器迁移与实验 workflow

本文件是接续实施说明。新服务器 pull 后，从第 1 步开始执行。
本地已迁移方法核心和两个 CUDA kernel，并实现通用校准入口、完整部署导出、
直接 packed INT4 加载和评测工具。真机训练配置、数据 processor、模型无权重构建
以及机器人控制循环，需要在目标服务器接入已有真机代码。

**本地验证状态与命令结果见 [docs/VALIDATION.md](docs/VALIDATION.md)。**
CPU 测试通过不能替代目标 GPU kernel、完整模型和闭环任务验证。

## 0. 最终流程与边界

```text
真机训练配置 + 原始 FastWAMJoint ckpt + 训练数据统计
                   + 真机训练集采样
                             ↓
                标准校准 observation 文件
                             ↓
         [W4A4 RHT: 先旋转权重并安装输入变换]
                             ↓
       可微去噪 → 动作敏感度 + 激活 reservoir
                             ↓
          D / γ / per-call clipping 校准
                             ↓
                 calibration.pt
                             ↓ 与原始未旋转模型合并（CPU）
                 deployment.pt
                             ↓
        meta/CPU 构建结构 → 直接装载 packed INT4
                             ↓
          W4A8 / W4A4 原生推理 + 可选 live Graph
                             ↓
     离线数值和延迟验证 → 真机重复任务 → SR / speedup
```

- W4A8 与 W4A4 分别校准，分别保存结果。
- RHT 必须在敏感度和激活采集前安装；RHT 与未旋转缓存不能混用。
- 部署时直接使用已旋转、量化的权重，不能再次旋转整个模型。
- SR 来自真实任务结果；离线 replay、CPU smoke 和 kernel 测试不产生 SR。
- 本次没有复制旧 `outputs/`、模型、大数据或其他量化对比方法。
- `deployment.pt` 保存的是可直接加载的模型 tensor 状态，不包含模型 Python 源码、
  tokenizer 外部文件或机器人 SDK。

## 1. 准备目标机器环境

在已经能够加载真机 FastWAMJoint 的 Python/Conda 环境中工作。先复现原始 BF16
推理，再安装本包。环境需要匹配的 PyTorch、CUDA Toolkit/nvcc、C++ 编译器和 ninja。
本地源环境记录为 Python 3.10、PyTorch 2.7.1+cu128；这是已使用版本记录，不代表
任意更新版本已验证。`pyproject.toml` 不会安装 FastWAM、机器人 SDK 或恢复训练环境。

从仓库根目录执行：

```bash
python -m pip install -e '.[test]'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import fastwam_steerquant; print(fastwam_steerquant.__file__)'
```

当前 Bash 默认使用 PATH 中的 `python`，不再指向原服务器的 Conda 绝对路径。
需要覆盖时设置 `STEERQUANT_PYTHON=/your/environment/bin/python`。

模型 checkpoint、训练配置、统计文件、VAE/text/tokenizer 资产和训练数据单独传输，
不放入 Git。GitHub 上只需要本仓库源码和必要配置模板。

## 2. 编译、验证两个 kernel

```bash
bash bash/setup_kernels.sh
```

脚本按 `third_party/versions.lock.json` 拉取原工作区使用的 CUTLASS 和
fast-hadamard-transform commit，然后编译新命名空间的扩展。
不会复用原服务器的 `.so`。依赖目录已有其他版本或修改时会停止，避免覆盖。
W4A4 当前只需 fast-hadamard-transform 的头文件，不依赖单独的 Python FHT 扩展。

运行 GPU 回归测试：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_w4a8.py tests/test_wam_w4a4.py tests/test_rht.py tests/test_deployment.py tests/test_rollout_graph.py -ra
```

检查输出中 GPU 测试确实执行，而不是全部 skipped。
不同 GPU 使用物理设备选择 `CUDA_VISIBLE_DEVICES`；应用内部默认使用可见设备 0。
当前核支持范围需要核对真机几何：W4A8 输入宽度对齐 32、W4A4 对齐 256，输出宽度对齐 8；
RHT 特殊宽度使用 3072=12×256、14336=28×512 的组合，其余按已有支持的二次幂路径。
支持等长专家流以及两段 text/proprio 流，不自动支持任意流布局。

editable 安装会从仓库寻找依赖。若安装 wheel 或把依赖放在其他地方，显式设置：

```bash
export CUTLASS_PATH=/your/dependencies/cutlass
export FAST_HADAMARD_PATH=/your/dependencies/fast-hadamard-transform
export CUDA_HOME=/your/cuda/toolkit
```

## 3. 补齐真机信息（主要待完成部分）

先收集下表信息，再实现适配，不能沿用示例数值猜测：

| 信息 | 来源/用途 |
| --- | --- |
| 原始模型构造代码、FastWAM revision | 校准加载与无权重部署构建必须为同一架构 |
| 训练配置、原始 ckpt、统计文件 | 校准与部署的来源一致 |
| 相机数量、顺序、图像拼接、尺寸、像素范围 | 与训练和真机原始推理的 processor 一致 |
| proprio 维度/归一化、动作维度/表示/归一化 | 正确敏感度尺度和动作执行 |
| action horizon、视频帧数、潜帧数、流划分 | 配置 D/γ 使用的 stream layout |
| 去噪步数、sigma shift、CFG | 校准和部署必须相同；当前 CFG=1 |
| 单任务的成功判定、超时、初始状态、执行 chunk 长度 | 可比较的 SR 和速度协议 |

在 `local/` 创建真实配置和 bridge；该目录已被 gitignore 排除，避免上传本机数据路径。
复制 `configs/real_robot.example.json` 后，修改其中所有示例路径和架构值。
例子中的 600 sites、3 个视频流、2 个动作流、32 horizon 并非真机默认保证。
`assets` 下每个值都应是一个实际文件；相对路径按配置文件所在目录解析。
将影响模型/归一化的外部配置、统计和 checkpoint 都列入 `assets`，用于恢复一致性检查。
大文件目前以路径/大小/mtime 检查；需要内容级溯源时，把其 SHA256 记录进 provenance。

```bash
mkdir -p local
cp configs/real_robot.example.json local/real_robot.json
export PYTHONPATH="$PWD/local${PYTHONPATH:+:$PYTHONPATH}"
```

配置的 `adapter` 为 `robot_bridge:make_adapter`。在 `local/robot_bridge.py` 中提供
`make_adapter(config)`，返回 `CallbackAdapter` 或实现相同方法的对象。
这是需要接入实际真机代码的文件，不是仓库中假装可运行的空白占位。

### 3.1 适配器方法契约

| 方法 | 必须实现的行为 |
| --- | --- |
| `load_model(*, device)` | 加载原始训练 ckpt，返回 eval 模型。尊重 cpu/cuda，不预先量化或旋转 |
| `build_model(model_config, *, device)` | 只构建完整模型结构，不加载原始 ckpt。支持 meta，或使用明确的 CPU 备用路径 |
| `prepare_inputs(model, record, *, seed)` | 返回可微去噪所需五个 tensor，语义与生产推理一致 |
| `infer_kwargs(model, record, *, seed)` | 真机原始 processor 处理 observation，返回 `model.infer_action(**kwargs)` 的参数 |
| `action_scale()` | 返回模型归一化动作坐标系中的标准差向量，正数且有限，维度等于 action_dim |
| `finalize_model(model, *, device)`（可选） | 修复模型的 device 属性、重建非 state_dict tensor/cache，接入 tokenizer 等外部资产 |
| `dataset_records()`（可选） | 迭代真机训练集样本，供 prepare_calibration_data.py 采样 |

可以把已有函数直接接到 `CallbackAdapter`，无须重复实现数据 loader 或机器人 SDK。
factory 本身应尽量只建立配置和 callback，不能为了创建 adapter 就加载原始 GPU 模型。

`prepare_inputs` 的返回值严格为：

```text
latents_video
latents_action
first_frame_latents
context
context_mask
```

当前校准每条记录使用 batch size 1。输入准备的 VAE/text 编码不求梯度；随后
`differentiable_joint_denoise` 会为采样的输入开启梯度，并得到最终动作敏感度。
核对真机模型仍暴露 `_predict_joint_noise`、`_joint_denoise_core` 及现有 scheduler API。
若真机分支改过去噪过程，需要对照修改 `state.py`，并在同一输入/seed 下验证
可微轨迹与生产轨迹输出一致。

如果沿用当前 FastWAM API，可复用
`adapters.fastwam.prepare_from_infer_kwargs(model, kwargs, seed=seed)`：

- `kwargs` 先由原始真机 processor 生成；图像为 `[1,3,H,W]`、浮点 `[-1,1]`，proprio 已归一化。
- 当前 helper 使用 prompt、CPU noise RNG、`tiled=False`、CFG=1；使用缓存 context、
  无 proprio 或其他输入方案时，按实际推理入口实现自己的 `prepare_inputs`。
- 与当前生产代码相同，视频噪声和动作噪声分别使用同一 seed 初始化的两个 generator。
  不使用一个连续推进的 generator 替代这两个 generator。
- `infer_kwargs` 必须显式包含 `num_inference_steps`，并传入一致的 `sigma_shift` 和 seed。
- 若动作 min-max 归一化到 [-1,1]，归一化 std 为 `2*raw_std/(max-min)`；使用 z-score
  或其他动作表示时按真实训练定义处理。零方差维度需要明确约定，不能默默除零。

### 3.2 构建部署模型的要求

`build_model` 接收的是导出时保存的整份配置。它只用其中的架构部分构建模型，
不能调用 `load_model` 或读取 `assets.source_checkpoint` 来初始化目标 Linear。
构建函数必须返回与训练模型相同的参数名、shape、bias 设置和模块结构。

推荐 meta 构建；如果 FastWAM 构造器会立即加载 pretrained 权重或使用不支持 meta 的操作，
先拆出“纯结构构建”。也可以用 `--construct-device cpu`：CPU 临时 BF16 分配可能仍存在，
但 target Linear 会在 `model.to(cuda)` 之前替换，仍不会将完整 BF16 权重搬上 GPU。

导出保存完整 state_dict 和已注册的 nonpersistent buffers。普通 Python 属性中的 tensor、
tokenizer 文件、设备属性和某些 runtime cache 不属于 state_dict，需要由构建/最终化接口处理。
`finalize_model` 不得重新创建 BF16 目标 Linear 或加载原始全模型权重。
不要在加载 packed 模型后调用全局 `.half()`/`.bfloat16()`，这会把需要 FP32 的量化尺度也转换。

## 4. 准备真机训练集校准样本

每条记录至少包含唯一的 `observation_id` 和非负整数 `sample_index`，其余字段交由适配器定义。
推荐内嵌相机输入、proprio、任务指令以及 episode/frame 标识；只保存 Tensor 和普通
Python 值，NumPy 数组先转 torch.Tensor。不要用路径间接指向会变化的数据而不记录版本。
`save_observation_records` 会转成 CPU tensor，并检查 ID 重复和可移植类型。

```bash
python scripts/prepare_calibration_data.py \
  --config local/real_robot.json \
  --output outputs/real/calibration_observations.pt --limit 100
```

或者在已有 loader 中调用 `save_observation_records(records, path)`。
采样应覆盖任务阶段和实际 observation 分布；采样策略由 `dataset_records()` 明确实现。
校准样本只从训练数据选择，真实 SR 使用另外的重复闭环 trial。

## 5. 校准预检、smoke 和正式运行

先执行 CPU synthetic smoke，验证本包基本数据链路：

```bash
python examples/cpu_smoke.py
```

该例子会对 20 个小型 Linear 实际计算敏感度、校准、导出并验证直接 packed 加载，
生成 `outputs/cpu_smoke/deployment.pt`。这不是正式 FastWAM 模型。

先对真实输入做配置预检：

```bash
bash bash/calibrate_w4a8.sh \
  --config local/real_robot.json \
  --observations outputs/real/calibration_observations.pt \
  --output-dir outputs/real/w4a8 --preflight-only
```

此预检仅检查配置、资产标识和记录；不会验证完整模型或 kernel 显存。
用少量记录、少量 epochs 在**独立 smoke 输出目录**跑完真实模型链路，确认形状和显存后再正式校准。

```bash
CUDA_VISIBLE_DEVICES=0 bash bash/calibrate_w4a8.sh \
  --config local/real_robot.json \
  --observations outputs/real/calibration_observations.pt \
  --output-dir outputs/real/w4a8

CUDA_VISIBLE_DEVICES=0 bash bash/calibrate_w4a4_rht.sh \
  --config local/real_robot.json \
  --observations outputs/real/calibration_observations.pt \
  --output-dir outputs/real/w4a4_rht
```

默认：4 probes、128 cache rows/cell、D20/γ30、seed42、单卡、source-dtype cache、
优化版统计和已提交 observation 后的 worker recycling。两个实验顺序运行。
可通过 `--num-probes`、`--max-rows-per-cell`、`--d-epochs`、`--gamma-epochs`、
`--batch-size`、`--recycle-every` 调整。

校准仍需要 BF16 模型和求导中间量，其内存需求不同于 INT4 推理。目标 GPU/CPU RAM 不足时，
先通过已有 worker 的 offload/回收选项定位阶段，不能以部署权重大小推断校准一定能运行。

```text
outputs/real/w4a8/
├── run.json                  输入/配置与恢复校验
├── sensitivity/
│   ├── sensitivity.pt
│   ├── activation_cache.pt
│   ├── manifest.json
│   └── timings.jsonl
├── calibration.parts/        各 Linear 的恢复进度/配置
├── calibration.pt            完整量化校准结果
└── calibration.json          来源信息，部署导出需要
```

中断后使用相同命令恢复。改变数据、训练权重、配置、位宽、旋转或优化预算时用新目录。
不要把 RHT 与未旋转、W4A8 与 W4A4 的缓存和恢复状态混用。
旧仓库的 checkpoint 格式和 observations 未自动兼容；本次应重新校准真机 ckpt。

## 6. 导出完整部署 checkpoint（CPU）

```bash
python scripts/export_deployment.py \
  --config local/real_robot.json \
  --calibration outputs/real/w4a8/calibration.pt \
  --output outputs/real/w4a8/deployment.pt

python scripts/export_deployment.py \
  --config local/real_robot.json \
  --calibration outputs/real/w4a4_rht/calibration.pt \
  --output outputs/real/w4a4_rht/deployment.pt
```

导出检查 calibration.json 与当前配置/资产身份一致，要求校准覆盖全部目标 Linear。
导出时加载原始未旋转模型，目标权重改存为 `uint8 [N,K/2]`：一个字节两个 signed INT4。
同时保存未量化权重、bias、FP32 scales、inverse D、γ、每步 activation scale 和 RHT signs。
已有 deployment 文件不会静默覆盖。

加载部署文件不需要原始 ckpt 的 tensor，但仍需要同版本模型代码、架构构建和外部资产。
校准/导出时需要的原始 ckpt 可另存，不进入真机推理 GPU。

## 7. 真机模型数值和显存验收

先对至少两个不同 observation 比较“原始模型 + 原生后端替换”和“直接 packed 加载”：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate_deployment.py \
  --config local/real_robot.json \
  --observations outputs/real/calibration_observations.pt \
  --calibration outputs/real/w4a8/calibration.pt \
  --deployment outputs/real/w4a8/deployment.pt \
  --output outputs/real/w4a8/validation.json
```

两个加载方式在独立进程运行，避免同时常驻两个模型。要求完整输出逐位一致；失败时定位
随机种子、processor、非 tensor 模型属性或 builder 差异，不先放宽误差阈值。
W4A4 RHT 对应改为其路径并执行同样的验证。

然后另存输出，以 `--cuda-graph` 重复验证。当前 Graph 首次捕获会比较 eager/Graph 输出，
后续 replay 复制当前输入。固定布局、去噪步数、CFG=1；不支持并发/重入。
首次 capture 必须在正式计时前完成，发生输入布局变化时明确报错。

另外比较量化参考后端与原生 kernel（GPU 测试包含算子 oracle）；真实模型输出误差与任务
质量需要单独检查。直接加载等价只证明加载方式一致，不能证明量化没有质量损失。

现有 kernel 的显存表仅是 DiT 范围，不能代替整机部署测试。分别记录：

- 加载阶段 peak：原始 BF16 是否进入 GPU。
- 稳定推理 peak：packed 权重、其他模块、激活、workspace。
- Graph 首次 capture/常驻内存：额外计入 VAE/text 和 Graph pools 后，是否适合目标卡。

不预设 24 GB 4090 一定能容纳所有全模型 Graph 配置；以真实模型数据为准。

## 8. 匹配条件测速

每种模式使用独立进程，同一配置、同一 observation 文件、相同 seeds、warmup/repeats。
主计时范围是 `infer_kwargs`（包括适配器预处理）+ 完整 `infer_action`；不含机器人 IO。
同步后计时，包含 GPU 工作完成。原始模型默认基线为 BF16，适配器需实际加载 BF16。
这里不把固定 DiT replay 时间当成完整真机推理时间。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --config local/real_robot.json --observations outputs/real/calibration_observations.pt \
  --output outputs/real/bf16_latency.json --warmup 3 --repeats 20 --limit 5

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --config local/real_robot.json --observations outputs/real/calibration_observations.pt \
  --deployment outputs/real/w4a8/deployment.pt \
  --output outputs/real/w4a8_latency.json --warmup 3 --repeats 20 --limit 5

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_inference.py \
  --config local/real_robot.json --observations outputs/real/calibration_observations.pt \
  --deployment outputs/real/w4a4_rht/deployment.pt \
  --output outputs/real/w4a4_latency.json --warmup 3 --repeats 20 --limit 5

python scripts/summarize_results.py latency \
  outputs/real/bf16_latency.json outputs/real/w4a8_latency.json outputs/real/w4a4_latency.json \
  --output outputs/real/speedup.json
```

汇总会拒绝协议、GPU 型号或软件版本不同的输入。主 speedup 使用 BF16 平均延迟 / 量化平均延迟，
另外给出中位数比例。若测试 Graph，三个命令都增加 `--cuda-graph` 并另存结果。
warmup/capture 总时间、加载峰值、稳定 allocated/reserved 峰值分别记录。
同型号两张卡的功耗/时钟/其他负载可能不同；正式比较应使用同一块空闲 GPU。

## 9. 接入已有真机控制循环并测 SR

在已有控制程序中创建一次 policy，重复复用；不在每个 action chunk 重载模型：

```python
from fastwam_steerquant.adapters import load_adapter
from fastwam_steerquant.policy import QuantizedPolicy
from fastwam_steerquant.evaluation import TrialRecorder

adapter = load_adapter("local/real_robot.json")
policy = QuantizedPolicy(
    "outputs/real/w4a8/deployment.pt", adapter,
    device="cuda:0", construct_device="meta", cuda_graph=False,
)
recorder = TrialRecorder(
    "outputs/real/w4a8_trials.jsonl", mode="w4a8",
    protocol_id="task1_protocol_v1", checkpoint_id="YOUR_DEPLOYMENT_SHA256",
)
# 以下调用位置接入现有控制循环，record 来自当前相机/proprio/指令：
# prediction = policy.infer(record, seed=trial_seed)
# 使用原有动作反归一化、执行和反馈逻辑处理 prediction。
# trial 结束后 recorder.record(task=..., trial_id=..., success=True/False,
#     reason=..., latencies_ms=[...], seed=trial_seed)
```

上面注释部分依赖真实控制程序，不是已经实现的机器人驱动。
`policy.infer` 返回模型原有输出，未自动替你做动作反归一化或执行。
一次异常后 call tracker 会在下次推理前复位；同一 policy 禁止并发调用。

实验前固定：任务定义、初始状态/物体分布、单次尝试时限、执行 action chunk 长度、
成功判据、试验次数、种子和人工干预规则。BF16、W4A8、W4A4 使用同一协议。
每次尝试都记录，失败、超时、异常终止不能从分母中漏掉。
recorder 不自动判断成功，也不负责重置机器人/环境。

```bash
python scripts/summarize_results.py trials \
  outputs/real/bf16_trials.jsonl outputs/real/w4a8_trials.jsonl outputs/real/w4a4_trials.jsonl \
  --output outputs/real/sr_summary.json
```

汇总按任务、协议、模式和 ckpt 分组，拒绝重复 trial ID。
SR = successes / trials，并保留次数。机器人观测/通信/动作执行延迟需在原控制程序另外测量，
不要把模型延迟的 speedup 直接称为整任务完成时间 speedup。

## 10. 可选 block 融合补丁

`patches/fastwam_block_fusions.patch` 保存原工作区 FastWAM `mot.py` 的现有融合调用差异。
先在目标 FastWAM 仓库检查：

```bash
git -C /your/FastWAM apply --check /your/Fastwam-steerquant/patches/fastwam_block_fusions.patch
```

检查通过且确实需要启用时再 apply。默认部署不开启 block 融合，W4A8/W4A4 的 Linear
量化 producer 融合已经包含在 kernel 中。RHT W4A4 目前拒绝 AdaLN/gate block 融合；
不要把它与已经实现的 RHT→inverse D→γ→A4 packing 融合混淆。
不自动对目标 FastWAM 版本应用补丁。

## 11. 交付验收清单

- [ ] 目标环境的 BF16 真机模型可正常执行，版本和训练资产已记录。
- [ ] `robot_bridge` 数据/模型/归一化契约实现，校准轨迹与生产轨迹验证一致。
- [ ] W4A8 / RHT W4A4 原生 kernel GPU 测试执行通过。
- [ ] 两种精度各自完成正式校准，没有拿 smoke 结果当正式结果。
- [ ] CPU 导出完整部署文件；meta/CPU builder 不加载原始 GPU BF16 权重。
- [ ] 两个不同 observation 的直接加载与原生替换输出一致。
- [ ] 若使用 Graph，变化输入的 replay 和显存验证通过。
- [ ] BF16/W4A8/W4A4 的测速协议一致，并分别记录加载与推理峰值。
- [ ] 原有真机控制程序接入 policy，动作反归一化与执行策略保持一致。
- [ ] 每次 trial 结果完整记录，汇总 SR、试验次数和 speedup。

远程仓库为 `https://github.com/CathyyW/Fastwamjoint-quantization.git`。
发布前检查 `.gitignore`，只提交源码/文档和去除机器信息的配置。
