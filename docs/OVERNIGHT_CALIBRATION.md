# 下载后自动校准：夜间监管脚本

入口：`scripts/overnight_calibration.py`。冒烟已通过，正式校准尚未启动；现已准备独立正式运行计划。
本机计划：`local/overnight.json`；可提交的示例：`configs/overnight.example.json`。

后续状态更新见 [下载后输入复核](INPUT_AUDIT_20260923.md)：下载已完成、磁盘已清理，
已补部署统计关联、用户声明的训练来源及25帧选择；VAE 身份已确认。
历史颜色与训练统计来源仍未证实，当前只批准保留原通道、等权敏感度的探索性冒烟。
以下“本次检查”数字是脚本创建时的历史快照，当前状态以只读检查为准。

## 当前冒烟计划（2026-09-23）

`local/smoke_only.json` 设置 `mode=smoke_only`，串行处理 pack、stack：
文件校验 → 输入检查 → 抽取25 obs → BF16 单条推理/准备路径对齐 →
W4A8 单 obs 梯度测试 → W4A4-RHT 单 obs 梯度测试。
梯度测试仅用 P=1、每 cell 最多16行；不是正式的25 obs、P=4校准。
此模式不运行 kernel 测试、正式校准、导出或真机程序。

在 tmux 窗口内执行：

```bash
cd /root/autodl-tmp/Fastwam-steerquant
CUDA_HOME=/usr/local/cuda-12.8 PATH="/usr/local/cuda-12.8/bin:$PATH" \
  /root/miniconda3/envs/qi-quant/bin/python scripts/overnight_calibration.py \
  --plan local/smoke_only.json --run
```

状态与阶段日志保存在 `outputs/real_robot_smoke_as_stored_v1/`。
此计划的 `COMPLETE` 仅表示冒烟通过，不表示已获得量化模型。
任务配置的 `provenance.formal_calibration_approved=false` 会阻止正式校准入口；
正式运行前必须另行决定实验假设与检查结果是否足够，不得只为过检查删除此保护。

## 冒烟通过后的校准/导出命令（2026-09-23）

两任务 BF16 输出有限、形状32×14、校准前向与 infer 逐位一致；四组单 obs/P=1
敏感度产物均通过600 sites、10 calls以及有限非负数值检查。整轮约11分钟。
用户随后要求校准/导出命令，已建立独立 `local/real_{pack,stack}_calibration_v1.json`
与 `local/calibration_as_stored_v1.json`，不修改旧冒烟配置或覆盖旧结果。
新配置批准沿用原通道与等权敏感度的实验校准，未将未知训练来源标成已验证。

在现有 tmux 的空闲 shell 中执行（本次只准备命令，没有启动）：

```bash
cd /root/autodl-tmp/Fastwam-steerquant
CUDA_HOME=/usr/local/cuda-12.8 PATH="/usr/local/cuda-12.8/bin:$PATH" \
  /root/miniconda3/envs/qi-quant/bin/python scripts/overnight_calibration.py \
  --plan local/calibration_as_stored_v1.json --run
```

运行先检查输入与 kernel，再依次执行 pack/W4A8、pack/W4A4-RHT、stack/W4A8、stack/W4A4-RHT。
每组正式参数为25 obs、P=4、D20 epochs、gamma30 epochs、rows128、batch128。
新运行会重新执行冒烟阶段，不复用不同配置身份的旧冒烟缓存。
每组校准后自动导出及两条 obs 的直接加载等价验证；不启动真机服务。

输出根目录 `outputs/real_robot_calibration_as_stored_v1/`，产物为
`{pack,stack}/{w4a8,w4a4_rht}/deployment.pt`，状态为根目录的 `status.json`。
当前剩余约48 GiB仅满足首组启动预算，不保证能同时保留四组结果。
保留40 GiB校准启动门槛、16 GiB导出门槛及3 GiB运行保留空间，空间不足停止且不自动删缓存。

## 做什么

1. 等待清单中的两份 ckpt 和 6 条轨迹出现正式文件名；`.part` 不算完成。
2. 核对每个文件的大小和 SHA256，不删除、覆盖或重下载校验失败文件。
3. 检查两个任务的人工确认信息、action_scale、抽样清单和磁盘空间。
4. 在独立进程核对 GPU、真实 adapter 和 25 obs，运行 W4A8/W4A4/RHT kernel 测试。
5. pack：抽取 25 obs → BF16 单条 forward 与校准准备路径逐位对齐测试。
6. pack W4A8：单 obs、P=1 的梯度 smoke → 正式校准 → 导出 deployment.pt → 两条 obs 的直接加载等价验证。
7. pack W4A4-RHT：同上，然后按同样顺序处理 stack。GPU 阶段全部串行。

正式校准固定 25 obs、P=4、D epochs=20、gamma epochs=30、rows=128、batch=128、seed=42。
每处理两个 obs 使用已有的安全提交/worker 回收机制；现有 worker supervisor 对外部 SIGKILL
仍有最多两次恢复重试。其他失败不会被夜间脚本忽略或跳过。
当前夜间流程不自动测试 CUDA Graph、测 speedup 或操作机器人；COMPLETE 只代表离线 eager 流程完成。

## 先检查，不运行

```bash
cd /root/autodl-tmp/Fastwam-steerquant
/root/miniconda3/envs/qi-quant/bin/python scripts/overnight_calibration.py \
  --plan local/overnight.json
```

默认模式只输出现状，不创建运行目录、不启动模型、不等待、不校准。
输出包含待下载文件、缺失确认项和下载结束后的预计剩余空间。

本次检查仍缺两任务的确认说明、归一化动作标准差和审核后的抽样清单。
**脚本不会替你填写这些字段，也不会将未知的 RGB/BGR、训练划分或统计绑定视为通过。**
填写方式见 [真实任务 workflow](REAL_ROBOT_CALIBRATION_WORKFLOW.md)。在启动前完成修改，
不要在脚本运行中改配置/源码/抽样清单；它会检测变化并停止。

## 后台启动命令（需要你决定启动时再执行）

```bash
cd /root/autodl-tmp/Fastwam-steerquant
CUDA_HOME=/usr/local/cuda-12.8 PATH="/usr/local/cuda-12.8/bin:$PATH" \
  /root/miniconda3/envs/qi-quant/bin/python scripts/overnight_calibration.py \
  --plan local/overnight.json --run --detach
```

不加 `--detach` 则在前台运行。脱离终端的后台进程不依赖本轮对话继续进行，
但仍会受服务器关机、平台回收或系统杀进程影响，不是跨服务器重启的系统服务。
本脚本不启动/停止现有下载器；下载器仍独立运行。

启动打印 PID 只代表已创建进程；立即查看 status 和 supervisor.log，确认未遇到锁冲突/启动错误。
同一输出目录及同一 GPU 的本脚本任务受文件锁保护，不允许重复运行；这不会锁住其他用户的 GPU 程序。

## 查看状态与停止

```bash
cat outputs/overnight/status.json
tail -f outputs/overnight/supervisor.log
tail -f outputs/overnight/pack_w4a8_calibrate.log
```

只有到相应阶段，阶段日志才会出现。

| 状态 | 含义 |
|---|---|
| WAITING_DOWNLOADS / VERIFYING_DOWNLOAD | 等待下载 / 正在校验 |
| RUNNING | 当前阶段运行中，记录子进程 PID 和耗时 |
| STAGE_COMPLETE / SKIPPED_VERIFIED_STAGE | 阶段完成 / 重启后复用通过校验的阶段 |
| BLOCKED | 缺确认信息、停滞/超时、空间不足、输入变化或资产不匹配 |
| FAILED | 子进程非零退出或其他异常；查看对应阶段日志 |
| STOPPED | 收到正常停止信号 |
| COMPLETE | 全部选定任务的离线校准、导出和 eager 等价验证完成 |

`events.jsonl` 保留状态历史，`supervisor.pid` 记录 PID。
如需停止，先查看 PID 并用 `ps -p PID -o pid,args` 确认身份，再执行 `kill -TERM PID`。
监管脚本会终止自己创建的当前校准进程组，不会终止独立下载器。
不要优先使用 `kill -9`：它无法进行子进程清理；若被系统 SIGKILL，重启前先检查遗留子进程。

## 超时与磁盘保护

默认每 30 秒检查，下载最多等待 24 小时；30 分钟无文件大小变化则停止。
校验大文件时可能暂时没有新的状态行，不代表卡死。每个子阶段最多 12 小时。
这些阈值可在计划 JSON 中修改，但应在启动前修改。

`disk_gib` 默认：小阶段至少 5 GiB，正式校准前至少 40 GiB，导出前至少 16 GiB，
阶段运行期间低于 3 GiB 则终止当前阶段。**这是保守的运行门槛，不是已经测得的真实模型空间需求。**
检查时预计下载结束后约剩 21.5 GiB，因此现有默认计划不会直接进入正式校准。
要先核算缓存/临时快照/导出体积并腾出空间，或在有依据后调整预算；脚本不自动清理任何文件。
每一组成功后会继续保留所有缓存和结果，后续组可能因为剩余空间不足而安全停止。
轮询不能保证避免所有突发磁盘/RAM 耗尽；底层原子保存及恢复检查仍是必要保护。

## 恢复与结果

问题解决后，用同一命令重新启动。已完成阶段只在源码/配置/命令一致、产物 SHA256
未变时复用；校准阶段通过原有 `--resume` 恢复。不要手改 `.done.json` 来绕过检查。
没有完成记录但已经存在的抽样/导出文件会被当作待人工检查的孤立产物，不自动覆盖。
如改变实验输入/源码，请选新的 output_dir，避免混用旧结果。

最终四份部署文件位于：

```text
outputs/overnight/pack/w4a8/deployment.pt
outputs/overnight/pack/w4a4_rht/deployment.pt
outputs/overnight/stack/w4a8/deployment.pt
outputs/overnight/stack/w4a4_rht/deployment.pt
```

单条 BF16 smoke 由 `scripts/overnight_preflight.py` 实现；VJP smoke 使用现有真实敏感度 worker。
只读检查和监管单元测试不加载真实模型、不启动正式校准，也不操作任何真机硬件。
