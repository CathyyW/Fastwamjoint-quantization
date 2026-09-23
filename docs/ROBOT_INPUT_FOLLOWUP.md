# 真机侧补充回复（用户提供截图）

## 已解决：VAE 身份

真机只读计算确认两个现用配置均指向：
`/home/agilex/World_Action_Model/physical_WM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`，
`redirect_common_files=false`。
SHA256 为 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`，与本地一致。
据此已填写两个本机任务配置的 `calibration_approval.vae_identity`。

## 尚未解决：历史 HDF 通道语义

采集代码 `imgmsg_to_cv2(..., 'passthrough')` 后直接保存 HDF，没有换色或记录 encoding；
当前 UVC MJPEG 普通分支可发布 rgb8，但不能证明历史数据使用了该分支。
原转换器直接 `PIL.Image.fromarray(raw)`，没有 BGR→RGB，再经过 JPEG/视频编码。
若历史数据实际是 BGR，事后擅自翻转会改变训练转换器看到的通道顺序。

可供实验负责人决定的方案：保留原 HDF 通道顺序，经现用在线尺寸/数值预处理作离线试验，
明确未复现有损转码，也不声称历史通道已证明为 RGB。若选择该方案，应增加显式 as_stored 约定，
不能为了通过现有检查直接填“RGB 已确认”。用户后续要求按此流程启动冒烟，
现已在本地配置和数据入口实施显式 `as_stored`，仅供探索性冒烟。

## 尚未解决：训练统计及敏感度尺度

真机未找到将现用 stats 绑定至两个 Joint checkpoint 的原始训练记录，
也未找到 z-score + clip[-5,5] 后的14维训练动作标准差。
stack 维10最小归一化端点约 -5.04313965，维12最大端点约 +5.23176041。
现有汇总不能精确恢复裁剪后的 std。

可供实验负责人决定的方案：仍沿用部署的输入/输出 normalizer，仅将 VJP 敏感度的 action_scale
设为14维全1，作为归一化动作坐标下的等权实验方案；它不是经验证的裁剪后训练 std。
用户后续要求按此流程启动冒烟，现已将本地 `normalized_action_scale` 明确设为等权全1，
并记录它不是训练标准差；原 `dataset_stats` 不变。

本次批准仅覆盖 BF16 单条推理与梯度冒烟，见 `local/smoke_only.json`。
正式校准仍未批准，配置中保留 `formal_calibration_approved=false` 保护；原交接包保持不变。
