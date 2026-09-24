# Pack / Stack 离线校准适配接续说明

## 1. 当前边界

本服务器负责离线校准、导出和离线验证；目标服务器负责 RPC、相机、RTC、动作执行及 SR。
本次没有启动校准、下载新权重、操作机器人或 push GitHub。

已接收 `/root/autodl-tmp/fastwam_calibration_handoff.tar.gz`：

- 压缩包 SHA256：`4408411a40875b858a4a6a61207e4b6e4cb1654a317545810a241500d2d6b185`。
- 解压目录：`/root/autodl-tmp/robot-handoff.oZZJ0y/fastwam_calibration_handoff`。
- 64 项文件校验通过；清单 SHA256：`a429d5d2c263e988d31d20181a5621210c973c14feaf5489d700a12af9ffc359`。
- 外部源码仍保持原样、独立存放。原包中旧量化代码只是上游 import 闭包，不替代 SteerQuant backend/kernel。
- `local/real_pack.json`、`local/real_stack.json` 已生成；属于本机配置，Git 忽略，不上传机器路径和资产。

新增代码不是占位：

| 文件 | 职责 |
|---|---|
| `adapters/real_robot.py` | 校验/导入源码，真实 processor、文本缓存、推理参数、动作反归一化，严格加载原始权重，纯 CPU 结构构建 |
| `adapters/robot_core.py` | 与原模型 dense 路径相同的可微联合去噪核心，校准和部署共享 |
| `adapters/robot_dataset.py` | 从已审核的 3 条 HDF 轨迹读取 9+8+8 帧，验证 SHA256、帧号、相机布局和颜色约定 |
| `scripts/configure_real_robot.py` | 从交接包生成每任务配置；不会把未确认项自动标为完成 |

此外已适配缓存 context 输入及部署共享模块别名。部署 payload 对每个别名都保存 packed buffer 引用，
不残留目标 Linear 的 BF16 别名权重。目标量化层以外的 VAE、投影、归一化等仍为浮点，
因此**整体 peak memory 不等于纯 INT4 权重大小**。

## 2. 正式校准前仍需确认

交接包 `tasks.json` 明确标记训练划分及统计绑定尚未确认。不要仅因为文件齐了就填写通过。
在本机任务 JSON 的 `calibration_approval` 中填写有依据的说明：

1. `stats_checkpoint_binding`：这份统计确实是该 checkpoint 使用的统计，或明确批准沿用当前真机部署统计。
2. `training_split`：选中的轨迹来自该任务训练划分，而非验证/测试集。
3. `image_pipeline`：确认 HDF 的 RGB/BGR 顺序；确认直接取 raw HDF 再走在线预处理是否可接受。
   训练转换过程可能包含 JPEG/H264，当前读取器没有模拟有损编解码。
4. `vae_identity`：本地 VAE SHA256 与真机使用的 VAE 对应；结构匹配不等于权重身份一致。

`normalized_action_scale.values` 需要 14 维、模型归一化坐标下的训练动作标准差，
`evidence` 记录来源/计算方法。不能把原始动作 std 或反归一化比例直接填进来。

- 当前 z-score 后还有 `[-5,5]` 裁剪。
- pack：报告中 extrema 未越界；统计绑定确认后，可核对使用 `std / (std + 1e-8)`。
- stack：报告中动作维度 10、12（从 0 编号）存在越界，裁剪后标准差不能仅靠原始 mean/std 精确推回。
  需要训练分布裁剪后的统计，或由实验负责人明确批准并记录近似方案。代码不默认填全 1。
- 不要用这 25 个 obs / 3 条轨迹偷偷重算训练归一化统计。

截至本次检查，两份 checkpoint 仍是 `.part`，尚未完成；6 条轨迹也未全部下载完成。
下载进程当前不存在，日志不是实时进度。需续传并通过下载脚本的 SHA256 校验后才进入校准。

## 3. 环境与本机配置

本机验证使用 `/root/miniconda3/envs/qi-quant/bin/python`，PyTorch 2.7.1+cu128。
不要将本机已有其他 `qi` 项目加入 PYTHONPATH；适配器会从交接包引入一致的源码。
若进程已导入其他路径的 `qi`，会明确报错；使用新进程，而不是混用同名包。

模型依赖沿用交接包 README 的清单。当前环境已可导入模型、Hydra processor、torchvision 和 h5py，
不需要安装 ROS、相机 SDK 或机器人驱动。本仓库 `pip install -e '.[test]'` 本身不覆盖全部上游模型依赖。

新机器重新生成配置的示例（本机两个文件已存在，不要覆盖已审核配置）：

```bash
cd /root/autodl-tmp/Fastwam-steerquant
PYTHONPATH=src python scripts/configure_real_robot.py \
  --handoff /path/to/fastwam_calibration_handoff --task pack \
  --checkpoint /path/to/joint_pack/step_015000.pt \
  --vae /path/to/Wan2.2_VAE.pth --output local/real_pack.json
```

stack 改任务名、checkpoint 和输出文件。脚本嵌入模型/processor 配置及资产 hash，
不会使用 YAML 中的 pretrained 下载入口，也不需要 ActionDiT 预训练权重；
原始任务 ckpt 必须严格覆盖完整 MoT 和 proprio_encoder。

## 4. 25 obs 的选择清单

下载完成、核对训练归属和颜色后，创建 `local/pack_sampling.json`（stack 同理）。示意结构：

```json
{
  "task": "pack",
  "source_color_order": "RGB",
  "training_split_evidence": "填写实际证据，不要照抄示例",
  "image_pipeline_evidence": "填写颜色与 raw HDF/在线预处理对应关系的证据",
  "episodes": [
    {
      "path": "/path/to/episode_0.hdf5",
      "sha256": "文件校验值",
      "frames": [{"index": 0, "stage": "阶段说明；此数组需要9帧"}]
    },
    {
      "path": "/path/to/episode_35.hdf5",
      "sha256": "文件校验值",
      "frames": [{"index": 0, "stage": "此数组需要8帧"}]
    },
    {
      "path": "/path/to/episode_70.hdf5",
      "sha256": "文件校验值",
      "frames": [{"index": 0, "stage": "此数组需要8帧"}]
    }
  ]
}
```

以上只是 schema，不是可直接使用的抽样清单。审核覆盖开始、接近、抓取/移动、放置/堆叠等阶段，
每条帧号升序且不重复。把清单绝对路径填入任务配置 `dataset.sampling_manifest`，
也可列入 `assets.sampling_manifest` 以纳入运行输入身份。

```bash
PYTHONPATH=src python scripts/prepare_calibration_data.py \
  --config local/real_pack.json --output outputs/pack/observations.pt --limit 25
```

一条记录只读取该时刻的三张相机图与 14 维 qpos，不需要从 HDF 取未来视频作为校准输入。
`--limit 25` 在导出阶段限制记录数量；后续校准使用导出文件里的全部记录。
不要把 `calibration.pt`（仅量化参数）和下面的完整 `deployment.pt` 混淆。

## 5. 校准与导出（当前尚未执行）

先通过上面的证据检查、文件 hash、BF16 单条 forward/VJP smoke 与显存检查。
完成下载不是直接跳过 smoke 的理由。以下 pack 命令再对 stack 独立重复，不能混用两个任务的数据或统计。

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/calibrate.py \
  --config local/real_pack.json --observations outputs/pack/observations.pt \
  --output-dir outputs/pack/w4a8 --activation-bits 8 --rotation none \
  --num-probes 4 --max-rows-per-cell 128 --d-epochs 20 --gamma-epochs 30 \
  --batch-size 128 --recycle-every 2 --preflight-only
```

预检查通过后去掉 `--preflight-only` 才真正开始 W4A8；W4A4 使用独立目录：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/calibrate.py \
  --config local/real_pack.json --observations outputs/pack/observations.pt \
  --output-dir outputs/pack/w4a4_rht --activation-bits 4 --rotation rht \
  --rotation-seed 42 --num-probes 4 --max-rows-per-cell 128 \
  --d-epochs 20 --gamma-epochs 30 --batch-size 128 --recycle-every 2

PYTHONPATH=src python scripts/export_deployment.py \
  --config local/real_pack.json --calibration outputs/pack/w4a8/calibration.pt \
  --output outputs/pack/w4a8/deployment.pt
```

W4A4 的导出命令将两个目录名换成 `w4a4_rht`；stack 同理。
需要预留校准激活、量化 artifact、完整部署权重的磁盘空间，不要按 1.7 MB 交接包体积估算。
当前剩余空间不足以保证同时保留所有任务/精度的全套缓存，应先估算，再决定是否分批执行。

## 6. 离线验收与目标机迁移

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/validate_deployment.py \
  --config local/real_pack.json --observations outputs/pack/observations.pt \
  --calibration outputs/pack/w4a8/calibration.pt \
  --deployment outputs/pack/w4a8/deployment.pt \
  --construct-device cpu --limit 2 --output outputs/pack/w4a8/direct_validation.json
```

然后单独验证 `--cuda-graph`，再按主 MIGRATION_WORKFLOW 的测速步骤比较 BF16/W4A8/W4A4。
**本适配器所有 direct load / benchmark 都要 `--construct-device cpu`，不支持 meta。**
构建 CPU 浮点结构不读取原始 ckpt；600 个目标 Linear 在移到 GPU 之前替换成 packed INT4。
CPU RAM 仍有结构构建开销，不代表整机 RAM 也只占 INT4。

真机至少需要：本仓库代码和 kernel 依赖、同一小交接包、对应任务 `deployment.pt`、重定位后的 JSON。
无需为了直接量化推理复制训练 HDF、激活缓存、原始任务 ckpt、独立 VAE 权重或 ActionDiT 预训练文件。
VAE 已包含在完整部署文件中；统计及文本缓存仍随小包保留。

JSON 重定位时保留 `real_robot` 中的全部内容/hash，只改 `handoff_root` 及
`assets.training_config/dataset_stats/context_cache/handoff_manifest` 等本机路径。
不要用生成脚本重新“确认”统计；可以直接复制校准时 JSON 再改路径。
推理适配器不会访问旧 `assets.source_checkpoint/vae_checkpoint` 路径。

纯模型使用入口（不发送任何机器人动作）：

```python
from fastwam_steerquant.adapters import load_adapter
from fastwam_steerquant.policy import QuantizedPolicy

adapter = load_adapter("local/real_pack.json")
policy = QuantizedPolicy("/path/to/deployment.pt", adapter,
                         device="cuda:0",
                         runtime_profile="configs/real_robot_w4a8_runtime.json")
# record: task、prompt、images[三相机RGB uint8 HWC]、state[14维物理量]
# result = policy.infer(record, seed=42)
# actions = adapter.denormalize_actions(result["action"])  # [32,14]，仅反归一化一次
```

上面的runtime profile仅适用于pack/stack **W4A8**：tile64、CPU结构构建、CUDA Graph、额外block fusion关闭。
W4A4-RHT不能使用该配置。源checkpoint与校准参数不变。启动包装与验收见
[REAL_ROBOT_W4A8_RUNTIME.md](REAL_ROBOT_W4A8_RUNTIME.md)。

目标机的 `serve_policy.py` / RTC 协议接线仍需在那边完成，当前不是可直接替换原启动命令的 ROS 服务。
建议隔离量化推理进程，避免已导入的另一份 `qi` 与交接源码冲突；保留原 RPC schema、执行节奏、
动作安全约束和成功判定。首次仅离线/不下发动作验证，硬件启停/初始姿态由已有控制程序负责。
SR 必须通过真实重复任务统计，不能由 CPU 测试、离线输出或 kernel speedup 推断。

## 7. 可复现的源码契约测试

本次实际结果：**79 passed，65 skipped（GPU 测试）**。另完成未缩小模型的 CPU 构建检查：
600 个目标 Linear，1,649 个 MoT 状态张量，6,020,688,078 个 MoT 参数；
proprio 权重形状 `[4096,14]`，VAE 类型 `WanVideoVAE38`。
本地 VAE `strict=True` 加载通过，未加载任务 ckpt、未初始化 CUDA。
这证明结构/本地 VAE 格式匹配，不证明本地 VAE 与真机权重 hash 相同。

```bash
FASTWAM_ROBOT_HANDOFF=/path/to/fastwam_calibration_handoff \
PYTHONPATH=src OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
python -m pytest tests -q -o addopts=''
```

真实源码契约测试使用缩小的 DiT / stub VAE 验证纯构建、不读取 checkpoint、
原实现与适配核心逐位相等、跨去噪步骤反向传播，以及两任务真实统计/文本/相机预处理。
不设环境变量时跳过外部源码测试；CUDA 隐藏时跳过 kernel/GPU 测试。
这些测试不能替代真实 checkpoint 的 forward、校准质量、完整部署显存和真机闭环验收。
