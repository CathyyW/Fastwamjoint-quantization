# 已完成实验的缓存清理（2026-09-23）

用户明确要求清理不再需要的中间缓存，保留 calibration.pt 和 deployment.pt。
执行前确认没有运行中的校准进程，两份 pack W4A8 模型 SHA256 与阶段完成记录一致，
直接加载验证两条 obs 逐位一致。

仅删除以下五份普通文件（路径相对仓库根目录）：

- outputs/real_robot_smoke_as_stored_v1/pack/w4a8_vjp_smoke/activation_cache.pt（1574958087 bytes）
- outputs/real_robot_smoke_as_stored_v1/pack/w4a4_rht_vjp_smoke/activation_cache.pt（2238132487 bytes）
- outputs/real_robot_smoke_as_stored_v1/stack/w4a8_vjp_smoke/activation_cache.pt（1574958087 bytes）
- outputs/real_robot_smoke_as_stored_v1/stack/w4a4_rht_vjp_smoke/activation_cache.pt（2238132487 bytes）
- outputs/real_robot_calibration_as_stored_v1/pack/w4a8/sensitivity/activation_cache.pt（11574056247 bytes）

共19200237395 bytes，约17.88 GiB。未删除任何模型、敏感度结果、观测数据、轨迹、
配置、日志或完成记录；未修改源码和计划，也未自动重新启动校准。

保留模型及其清理前核验 SHA256：

- pack/w4a8/calibration.pt：769be832c94e654bfbc9fd68d979a91e4937e3655a805a5ef2678ad25659b36a
- pack/w4a8/deployment.pt：4e3b7ff923623117d394d267243e1bb62f3027f5920ad5413754129e8dcc95b8

上述模型位于 outputs/real_robot_calibration_as_stored_v1/。

## 恢复含义

删除是永久删除，没有回收站副本；如需这些激活缓存，必须从保留的输入重新采集。
旧独立 smoke_only 运行仅保留历史证据，不能再以原输出目录直接完成缓存完整性续跑，
需要重测时使用新目录，不得修改旧 done.json 伪装缓存仍存在。

当前正式运行的 pack 两份 VJP 冒烟缓存保持不动，原监管流程需要检查它们。
pack W4A8 校准阶段完成记录只依赖 calibration.pt 和 calibration.json，
因此删除其内部激活缓存不影响原监管流程跳过该已完成阶段。
后续重新调整 D/gamma 需要重建缓存；推理和已有部署文件不受影响。
