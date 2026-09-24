# Stack W4A4-RHT：75 obs 重校准

2026-09-24，仅离线校准/导出/验证，不启动机器人。保留原 25 obs 模型。

## 实验约束

- 原 checkpoint：joint_stack/step_015000.pt，SHA256 `a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f`。
- episode 0/35/70 各取 25 帧，总计 75 obs。原 25 帧全部保留，并保持原 sample_index/随机种子；新增帧按最大时间间隔补齐，平局取较早帧。三份预览已逐张检查。
- 排除原六帧误差测试集及各帧前后 3 帧。测试仍来自相同三条轨迹，且已用于旧模型分析，因此只能作为固定诊断集，不能视为独立泛化测试。
- P=4，max_rows_per_cell=128，D epochs=20，gamma epochs=30，batch_size=128，seed/rotation_seed=42，recycle_every=2；保持旧设置。
- 保持 as_stored 图像、原 VAE、现用统计、归一化动作尺度等实验假设。扩大 obs 不补足历史训练归属/颜色来源证据，也不保证误差下降。

## 本地文件

- `local/prepare_stack75.py`：构造嵌套抽样与观测文件；已执行，禁止覆盖已有结果。
- `local/real_stack_calibration_75obs_v1.json`、`local/stack_sampling_75obs_v1.json`：新配置与抽样清单。
- `local/run_stack75.py`、`local/stack_w4a4_75obs_plan.json`：检查输入哈希、下载完整性、磁盘余量并串行执行各阶段。
- `local/evaluate_stack75.py`：复用旧六帧 BF16 基线，使用新部署模型实际 native W4A4-RHT forward，输出 25/75 obs 对比。
- `src/fastwam_steerquant/adapters/robot_dataset.py`：允许显式 per_episode_counts，默认仍为旧 9+8+8，并在读取前拒绝选中 excluded_observation_ids。

这些 local 配置/脚本含本机路径，默认不由 Git 跟踪；迁移实验时需另行携带，不代表已推送远程。

## 自动工作流

校验输入 → 75 obs preflight → 重新采集敏感度及 D/gamma 校准 → 导出 → 两 obs 直接加载等价性验证 → 固定六 obs 误差对比。

输出根目录：`outputs/stack_w4a4_rht_75obs_v1`。

- `deployment.pt`：最终部署模型（只有导出成功后才存在）。
- `direct_validation.json`：加载/forward 验证结果。
- `evaluation/COMPARISON_25_VS_75.md`：归一化动作、关节角、夹爪误差对比。
- `status.json`、`events.jsonl`、各阶段 `.log`：状态和故障原因。
- `sampling_audit.json`：旧 25 帧内容/编号保持及评估帧排除检查。

异常停止，不删除文件；完成阶段按身份与输出哈希跳过，校准阶段由既有 pipeline 恢复。导出或评估若留下孤立输出需人工检查，脚本不会覆盖。运行期间不要修改 src/scripts/tests、固定输入或配置，否则阶段边界检查会停止。

## 磁盘门槛

本实验将通用 40 GiB 门槛改为独立配置的 31 GiB。旧 25-obs 激活缓存实测约 10.78 GiB；13200 个 cell 已达 128 行上限，只有1200个本体状态 cell 从25行增至75行，预计新缓存约11.01 GiB，并非三倍。按原子快照双份缓存及校准/导出产物估算约22 GiB峰值新增占用，31 GiB含约9 GiB余量。启动前约35.56 GiB可用；运行中低于4 GiB自动停止，导出前要求16 GiB。估算不是空间保证；没有删除旧结果。

## 查看与恢复

```bash
tmux select-window -t pack-calib:stack-w4a4-75obs
tail -f /root/autodl-tmp/Fastwam-steerquant/outputs/stack_w4a4_rht_75obs_v1/calibrate.log
```

启动脚本使用 qi-quant 环境、CUDA 12.8、GPU 0，以及原仓库的 CUTLASS/fast-hadamard-transform 源码路径。断开 SSH 不影响 tmux 内进程；服务器关机/重启仍会停止任务。

预计约6–8小时，实际以敏感度阶段逐 obs 进度为准。完成后的离线误差不等于真机成功率，也不是 speedup 测试。
