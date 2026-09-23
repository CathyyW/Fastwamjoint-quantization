# 下载完成后的本地输入复核（2026-09-23）

## 结论

不是缺少交接 tar.gz，不需要再次发送整包。原包已具备模型源码、实际部署配置、统计、文本缓存。
本次重新流式核验了两份 ckpt、六条 HDF 和本地 VAE；两份 ckpt 都在真实、完整的模型结构上
通过 `mot.load_state_dict(strict=True)` 与 `proprio_encoder.load_state_dict(strict=True)`。
没有原始预训练 DiT/ActionDiT 读取，没有 forward、CUDA 初始化、校准、机器人动作或监管重启。

可复现入口：`scripts/audit_real_robot_inputs.py`。
实际证据：`local/input_audit_20260923/audit.json`；图像预览在其 `pack/`、`stack/` 子目录。
原始交接包、已下载权重和训练数据均未修改。

## 已闭合的证据

| 项目 | 本地复核结果 |
|---|---|
| pack ckpt 跨服务器身份 | SHA256 `2b5da8ac40a2701f65f1f72cd613713047271deafbd2da929b3ed8ab75b17a49`，与原机报告一致 |
| stack ckpt 跨服务器身份 | SHA256 `a86b2223d4b0f1fcb079532a5a2898a5ab9f529da3143994721cdaa32d29fc9f`，与原机报告一致 |
| 权重结构 | 两者均 step=15000，MoT 1649 张量、6,020,688,078 元素，BF16；严格 CPU 加载通过 |
| 统计文件身份 | 两份统计的哈希与原机记录、交接 provenance 一致；报告第2节明确它们分别是 pack/stack 服务配置使用的文件 |
| 文本条件 | 两任务缓存均可读取，context `[128,4096]`，原始 padding 置零后 mask 全 true，与生产语义一致 |
| 数据完整性 | 六条 HDF 均通过预先固定的大小/SHA256 校验 |
| 数据字段 | 三相机均为 `[N,480,640,3] uint8`，qpos/action 均为 `[N,14]`，值有限；根属性 compress=false、sim=false |

注意“当前部署配置搭配这份统计”和“它就是该 ckpt 训练 run 的统计”不是一回事。
前者已有文件证据，后者依然未由独立训练记录证明。后者不会被偷偷标成已验证。

用户此前明确指出这两个数据集是对应任务/ckpt 的训练集，并要求取三条轨迹。
本次将此记为**用户提供的训练来源声明**；不再重复要求用户证明其已经说明的仓库对应关系。
但不存在独立的 train/val 清单或 HDF→LeRobot 编号映射，若将来提供的记录与此冲突应重新抽样。

## 25 obs 选择已经准备

已查看六张三相机联系图，按时间分层候选帧标注了可见操作阶段。
pack 覆盖积木、胶带、红色物体依次入容器；stack 覆盖杯子操作、底座准备和上层杯放置。
阶段标签是图像审阅的解释，不是数据自带 ground truth。

| 任务/轨迹 | 总帧数 | 选中的帧号（从0开始） |
|---|---:|---|
| pack / 0 | 894 | 0,112,223,335,446,558,670,781,893 |
| pack / 35 | 863 | 0,123,246,369,493,616,739,862 |
| pack / 70 | 823 | 0,117,235,352,470,587,705,822 |
| stack / 0 | 553 | 0,69,138,207,276,345,414,483,552 |
| stack / 35 | 642 | 0,92,183,275,366,458,549,641 |
| stack / 70 | 644 | 0,92,184,276,367,459,551,643 |

文件：`local/pack_sampling.json`、`local/stack_sampling.json`，已接入两个本机任务配置。
每任务9+8+8=25帧。颜色尚未确认，因此 `source_color_order=null`，没有导出可正式校准的 obs.pt。
末帧可用于独立 observation 校准；本流程不读取未来 GT 动作/视频，不据此构造33帧训练窗口。

## 当前只剩三个类别需要补充或决定

### A. VAE 文件身份

本地 VAE：`/root/autodl-tmp/asset/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。
大小2,818,839,170字节，SHA256：

```text
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36
```

原机报告只给出同样的文件大小，明确没有算内容哈希；同大小不能证明同一权重。
只需真机补一个 SHA256 字符串，不需要上传 VAE。

### B. HDF 图像通道与处理约定

六条 HDF 均无 RGB/BGR 编码属性。按存储通道显示，pack 呈黄色容器、黄蓝积木、红色物体，
stack 呈粉/黄/蓝杯，视觉上合理，但不能单靠“看起来合理”证明存储顺序。
需要原采集代码/数据转换约定或操作者对实际物体颜色的确认。

本地计划从 raw HDF 读图、转换到 RGB 后调用同一在线预处理；不重演训练转换时可能存在的 JPEG/H264。
这与在线图像路径对齐的选择应在 image_pipeline_evidence 记录，不隐瞒有损转码差异。

### C. 动作敏感度用的 action_scale

这不是缺 normalizer 文件，也不需要改掉已有输入/输出归一化；缺的是 VJP 动作敏感度加权所用的
**训练分布经过模型归一化后的动作标准差**及来源/近似约定。

本地重算确认：现有统计的 pack extrema 无越界；stack 的0-based动作维10、12存在潜在裁剪。
若统计确属目标训练分布，pack 可以使用 `s/(s+1e-8)`；stack 的裁剪后 std 无法仅由原始 mean/std/min/max精确还原。
若批准截断前 std 近似，则可明确记录这种方法，不能声称它是精确裁剪后训练统计。

三条轨迹合计 pack 2580 帧、stack 1839 帧：输入状态及动作都没有触发当前统计的 `[-5,5]` 裁剪。
这只说明当前子集没有观察到越界，不能推出整个训练集没有裁剪。
子集归一化动作 std 大致 pack 0.695–1.193、stack 0.771–1.194；仅作为诊断写入 audit.json，
**没有回填到 normalized_action_scale，也没有重写 dataset_stats.json**。

## 本机配置修改的范围

- 已填 `calibration_approval.stats_checkpoint_binding`：依据是“现用部署关联”，明确不声称训练来源已证明。
- 已填 `calibration_approval.training_split`：依据是用户指定的训练数据来源，明确没有独立 split manifest。
- 已接入已审阅帧号/阶段的 sampling 文件；其 RGB/BGR 和图像处理证据仍为空。
- 已记录本次 ckpt 哈希一致、严格 CPU 加载通过和审计文件位置。
- `image_pipeline`、`vae_identity`、`normalized_action_scale` 仍保持待确认。

监管脚本仍未重启。补齐上述三类后才能继续；不是另要一套源码/统计/文本文件。

## 给真机侧的最小请求

只做只读核对，不启动模型、ROS 或机器人，不重新打包整套工程。请回复：

1. 当前配置实际指向的 Wan2.2_VAE.pth 的绝对路径及 `sha256sum` 结果。
   原报告路径为 `/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`；
   若当前路径不同，请以当前配置为准。
2. pack_3_objects_plus / stack_3_cups_gen 的原始 HDF 相机数组是 RGB 还是 BGR？
   给出采集/转换代码位置，或指出无法从现有代码证明。是否接受从 raw HDF 经 RGB 在线预处理作为校准输入？
3. 是否有两份现用 stats 对应的真实训练统计来源，特别是 stack 的归一化并clip后的动作 std？
   如没有，直接回答没有；由实验负责人决定是否采用明确标注的截断前 std 近似，不要编造统计。
