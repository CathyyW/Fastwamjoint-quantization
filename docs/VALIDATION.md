# 本地迁移验证记录

日期：2026-09-22。源码来自本机 `Fastwamjoint-quantization` 工作区。
原项目源码和实验产物未迁移覆盖；新包独立位于 `Fastwam-steerquant`。

## 环境与范围

- Python 3.10；PyTorch 2.7.1+cu128；pytest 9.1.1。
- 此会话中 `torch.cuda.is_available() == False`、device_count=0，nvidia-smi 无执行权限。
- cgroup memory.max 为 2 GiB。多个检查并行期间整套 pytest 曾以 137 退出，未返回
  Python 异常；没有据此宣称整套通过。随后将所有测试拆成串行组执行。
- 未在此会话运行真实 FastWAM 全模型校准、编译 CUDA 扩展或启动机器人。

## 已通过验证

1. CPU 核心回归：拓扑、stream、敏感度、缓存、D/γ 校准、checkpoint、断点恢复。
2. RHT 重命名的源码数值对照：宽度 256、1024、3072、4096、14336；随机符号、
   前向和反向与原 `official` 变换均逐位一致（atol=rtol=0）。
3. 部署格式：W4A8、W4A4、RHT W4A4，meta 和 CPU 两种 builder；直接加载的每个
   packed buffer 与原生替换路径完全一致，未量化层和 nonpersistent buffer 正确保留。
4. 错误检查：部分校准导出、重复输出文件、非 packed 权重、构建 shape 错误、
   校准配置变更、重复 observation/trial 等情况会拒绝执行。
5. CPU smoke：实际执行 20 个小型 Linear 的敏感度→校准→导出→meta 直接加载，
   并比较所有部署 buffer。生成 `outputs/cpu_smoke/deployment.pt`（已 gitignore）。
6. 十个 CLI 的导入和 `--help` 成功；三个 Bash 脚本通过 `bash -n`。
7. wheel 构建成功，检查包含两个 kernel 的 6 个 `.cu/.cpp/.cuh` 文件，未打包对比方法。
8. W4A8 的 `.cu` 和 `binding.cpp` 与原文件完全相同；W4A4 的两个主 CUDA/C++ 文件
   与原文件进行 RHT 命名替换后的文本完全相同，未重写 kernel 数学/调度设计。

## 串行测试结果

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/test_adapters_evaluation.py tests/test_deployment.py tests/test_recovery.py -ra
# 19 passed, 2 skipped

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest tests/test_pipeline.py tests/test_rht.py tests/test_async_offload.py \
  tests/test_sensitivity_optimization.py tests/test_rollout_graph.py \
  tests/test_w4a8.py tests/test_wam_w4a4.py -ra
# 21 passed, 63 skipped

PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest tests/test_policy.py -ra
# 2 passed
```

以上三组合计 **42 passed、65 skipped**，跳过项均要求 CUDA。
policy 测试覆盖多次 action chunk 的状态复位、异常恢复、配置不匹配和重入拒绝。

## 必须在目标 4090 补验

- 固定版本依赖和扩展编译；真实目标几何的 W4A8/W4A4/RHT kernel 测试。
- 真机 processor、动作归一化、噪声准备、可微轨迹与生产轨迹对齐。
- 真机 FastWAM 的无权重 builder：CPU/meta 支持、设备属性和外部资产恢复。
- 完整模型直接加载与原生替换的输出逐位一致。
- 多 observation、后续 action chunk 和可选 Graph 的变化输入 replay。
- 真实加载峰值、稳定推理峰值与 Graph 内存，不能用 tiny CPU smoke 估算。
- 闭环 SR 与匹配条件 speedup。

## 迁移实施说明

保留原核心文件布局，新增 deployment/adapters/policy/evaluation 模块；没有为规划中的
每个概念建立空文件。RHT 命名用于公开旋转配置、Python 函数和 CUDA 绑定。
来源注释记录 RHT 相关文件、原始 commit 和许可证信息。

新的校准格式为 `fastwam_steerquant_calibration_v1` / `fastwam_steerquant_calibration_rht_v1`，
完整部署格式为 `fastwam_steerquant_packed_deployment_v1`。没有静默转换原仓库旧校准缓存。
新的 FastWAM 输入 helper 对齐生产代码的独立视频/动作 generator；这项差异已由测试验证。

外部 block 融合 patch 基于 FastWAM commit
`7faa71108368fbb3b6885649f112af607427a2d4` 上的工作区改动提取，未自动应用。
依赖 pins 对应原工作区。FHT 原工作区有 setup.py 修改，但当前内联融合路径只使用头文件，
因此没有把该安装脚本修改或旧扩展二进制迁入新仓库。

具体接续步骤见仓库根目录的 [MIGRATION_WORKFLOW.md](../MIGRATION_WORKFLOW.md)。
