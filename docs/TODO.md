# TODO.md

## 当前架构目标

项目采用新架构：

```text
src/legacy/full_model/  # 旧大模型整体归档
src/models/             # MTL-Lite 新主线与通用模型模块
src/diagnostics/        # 独立诊断与可视化系统
```

核心原则：

- 旧大模型整体迁入 legacy，保留为可运行历史快照；
- legacy 只补 README 和边界说明，暂时不投入额外修复或重构；
- MTL-Lite 独立实现，不继承旧模型；
- 通用模块保留在主线位置；
- 诊断系统独立，不强耦合训练 forward；
- 每次只推进一个重构目标。

## 阶段 1：legacy 归档说明

- [x] 添加 `src/legacy/full_model/README.md`，说明旧模型边界、运行方式和维护策略
- [x] 明确 legacy 是旧大模型快照，不再作为新主线开发对象
- [x] 明确旧模型如需运行，应使用 legacy 快照自身的脚本和配置
- [x] 明确禁止提交 legacy 下的 `local_paths.yaml`、日志、权重和 checkpoint
- [x] 暂时放弃 legacy 内部 import 修复、runner 接入和额外重构，除非用户明确要求复现旧模型结果

## 阶段 2：MTL-Lite 新主线基础接口

- [x] 新增 `src/models/outputs.py`
- [x] 定义 `MTLLiteOutput`
- [x] 定义 `MTLLiteLosses`
- [x] 新增 `src/models/temporal/__init__.py`
- [x] 新增 `src/models/temporal/pooling.py`
- [x] 实现 `masked_mean_pool`

## 阶段 3：MTL-Lite 新模型骨架

- [ ] 设计后续 `src/models/temporal/encoders.py`
- [x] 新增 `src/models/mtl_lite.py`
- [x] 实现 `MTLLiteDepressionModel` 骨架
- [x] MTL-Lite 不继承 legacy full model
- [x] MTL-Lite 只依赖通用 backbone、task heads、metrics 和 temporal utilities
- [x] 不引入 contrastive、CGC、adaptive mask、PCGrad、LDS 或 `loss_dist`

## 阶段 4：MTL-Lite 测试

- [x] 新增 `tests/test_mtl_lite_forward.py`
- [x] 新增 `tests/test_mtl_lite_loss_backward.py`
- [x] 新增 `tests/test_mtl_lite_config.py`
- [x] 使用 dummy backbone，避免下载权重
- [x] 不读取真实数据
- [x] 不依赖 `configs/local_paths.yaml`
- [x] 检查 forward shape
- [x] 检查 regression head 梯度非零
- [x] 检查 loss/metric 尺度一致

## 阶段 5：配置与 baseline

- [x] 新增 `LOSSES` 配置约定
- [x] 新增 regression-only baseline override
- [x] 新增 MTL-Lite baseline override
- [x] 新增 MTL-Lite debug smoke override
- [ ] 保持 `configs/local_paths.yaml` 私有且不提交
- [ ] 服务器运行 MTL-Lite debug smoke

## 阶段 6：诊断系统独立化

- [x] 梳理当前 `src/utils/visualize.py`
- [x] 新增 `src/diagnostics/`
- [x] 新增 `src/diagnostics/regression.py`
- [x] 新增 `src/diagnostics/embeddings.py`
- [ ] 新增 `src/diagnostics/temporal.py`
- [x] 新增 `src/diagnostics/model_attention.py`
- [x] 新增 `src/diagnostics/reports.py`
- [x] 新增 `src/diagnostics/correlation.py`
- [x] 新增 `src/diagnostics/occlusion.py`
- [x] 新增 `src/diagnostics/keyframes.py`
- [x] 支持 `predictions.csv` 导出
- [x] 支持 prediction-target scatter
- [x] 支持 residual histogram
- [x] 支持 BDI 区间误差分析
- [x] 支持 severity group 误差分析
- [x] 支持 high-error / low-error subject ranking
- [x] 支持 t-SNE / UMAP
- [x] 支持 metrics / predictions 相关系数热力图
- [x] 支持 occlusion sensitivity 遮掩影响热力图
- [x] 支持 temporal occlusion 关键帧重要性热力图
- [x] 支持模型自身关注区域热力图（Grad-CAM 可用时优先，否则回退到 input-gradient）
- [ ] 服务器运行 MTL-Lite 离线诊断脚本
- [ ] 保留旧诊断入口的向后兼容
- [ ] 新增 Shortcut Audit 离线诊断入口，不参与训练 forward

## 阶段 7：实验路线

- [ ] 运行 regression-only baseline
- [ ] 运行 MTL-Lite baseline
- [ ] 在相同 split、seed、backbone、时序编码器和指标下比较二者
- [ ] 建立 OpenFace 质量、姿态、gaze、AU 与 BDI/误差的相关性诊断
- [x] 运行第一轮输入消融：aligned RGB、grayscale、blur、center_mask、boundary_erased
- [ ] 运行第二轮黑伪迹输入消融：black_to_gray、black_to_mean、black_to_blur、soft_center_mask、inner_crop_resize
- [ ] 建立 landmark-only temporal baseline
- [ ] 建立 AU / pose / gaze-only temporal baseline
- [ ] 建立 RGB + behavior late-fusion baseline
- [ ] 将辅助任务从单纯 BDI ordinal 扩展到 AU、landmark motion、pose/gaze 或 expression 相关行为任务
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + CCC loss 消融
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + LDS 消融
- [ ] 行为辅助任务稳定后，再考虑 MTL-Lite + `loss_dist` 消融
- [ ] 固定行为表征 baseline 稳定后，再考虑动态任务权重或梯度冲突处理

## 近期验证

- [ ] 添加 regression head 初始化测试，确保不使用训练标签统计量初始化预测参数
- [ ] 验证 `validation_step` 可以在 `no_grad` 下 forward，且不改变 metrics
- [ ] 确认 LDS label weighting 只使用训练集标签，不接触 val/test 标签
- [ ] 验证 `MODEL_WEIGHT_PATH` 能加载 raw `state_dict` 和 `{"state_dict": ...}` checkpoint
- [x] 让 `FREEZE_BACKBONE` / `FINETUNE_LAST_N_BLOCKS` 配置在 MTL-Lite 中真正控制 backbone 可训练范围
- [ ] 验证多 GPU 下 DDP metric logging 和 best-weight 保存行为
- [ ] 添加 bf16-mixed precision 下预测和 loss finite 测试

## 论文工具

- [ ] 标准化实验日志格式
- [ ] 添加将 `metrics.csv` 导出为 LaTeX 表格的脚本
- [ ] 将诊断图表输出组织为论文可用目录结构
- [ ] 维护 `docs/RESEARCH_NOTES.md`，记录 OpenFace、AVEC2014、面部行为建模、多任务学习和捷径学习相关论文

## OpenFace 行为表征研究路线

- [x] 建立非抑郁捷径验证框架设计文档 `docs/SHORTCUT_AUDIT_DESIGN.md`
- [ ] 确认当前数据使用的 OpenFace 版本、命令、输出字段、裁剪尺寸和帧采样方式
- [ ] 确认是否保留 OpenFace 原始 CSV，并列出可用字段：`confidence`、`success`、pose、gaze、AU、landmark
- [ ] 统计每个视频的 OpenFace `confidence` 均值、方差和低置信帧比例
- [ ] 统计每个视频的 `success` 失败帧比例
- [ ] 统计 pose/gaze 分布和 landmark 抖动
- [ ] 分析 OpenFace 质量变量与 BDI、预测误差、残差的相关性
- [ ] 对高误差 subject 生成 aligned face、attention、occlusion、keyframe case study 图组
- [ ] 验证模型关注区域是否集中于眼、眉、嘴、鼻唇沟，而不是脸部边界、头发、眼镜、黑边或裁剪伪影
- [ ] 建立 `E0_openface_quality_correlation` 实验记录模板
- [ ] 建立 `E1_input_ablation` 实验记录模板
- [ ] 建立 `E2_landmark_temporal_baseline` 实验记录模板
- [ ] 建立 `E3_au_pose_gaze_baseline` 实验记录模板
- [ ] 建立 `E4_rgb_behavior_late_fusion` 实验记录模板

## Shortcut Audit 实施路线

- [ ] 新增 `src/diagnostics/openface_quality.py`，读取 OpenFace CSV 并生成 subject-level quality summary
- [ ] 新增 `src/diagnostics/shortcut_audit.py`，合并 `predictions.csv`、OpenFace quality summary 和 split 信息
- [ ] 新增 `scripts/audit_shortcuts.py`，作为非抑郁捷径验证的离线入口
- [x] 修正 Shortcut Audit 合并键：优先使用完整 `video_id`，避免 Freeform/Northwind 与短 `subject_id` 错配
- [x] 修正 Shortcut Audit 的 `video_id` 规范化：兼容 `*_video` 与 `*_video_aligned`，确保 OpenFace summary 与 prediction CSV 能够按同一视频匹配
- [x] 重新运行 Shortcut Audit，并确认 `shortcut_audit_report.md` 中 `Matched samples` 等于当前预测样本数；若为 0 或明显偏低，不得解释 shortcut risk
- [x] 输出 `openface_quality_summary.csv`
- [x] 输出 `shortcut_correlation.csv`
- [x] 输出 `shortcut_correlation_heatmap.png`
- [x] 输出 residual vs confidence / pose / quality 诊断图
- [x] 输出 `shortcut_audit_report.md`
- [x] 实现 shortcut-only BDI predictor baseline：mean、linear regression、ridge、random forest
- [x] 在 shortcut-only predictor 中优先加入按 `subject_id` 分组的交叉验证，避免同一 subject 的 Freeform/Northwind 泄漏到不同折中
- [x] 将 grouped CV shortcut-only predictor 写入正式诊断输出，至少报告 mean baseline、RGB 模型、ridge 多个 alpha 的 MAE/RMSE/Pearson
- [x] 设计并接入输入消融配置 `DATASET.INPUT_VARIANT`，支持 `rgb`、`grayscale`、`blur`、`center_mask`、`boundary_erased`
- [x] 扩展黑填充/硬边界伪迹输入消融，支持 `black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`
- [x] 新增 `scripts/audit_black_artifacts.py`，离线统计 aligned frame 中黑像素、黑边界和硬边缘与预测误差的关系
- [ ] 将 `landmark_heatmap` 接入 OpenFace landmark CSV 或 behavior baseline 路径，不在 RGB dataset 中伪造 landmark 输入
- [ ] 设计区域级 attention/occlusion 统计：eye、brow、mouth、face center、boundary、non-face

## 当前实验诊断待办：预测压缩与捷径风险

- [x] 基于最新 `test_predictions.csv` 记录 regression 诊断：整体 MAE 约 8.91、RMSE 约 10.95、Pearson 约 0.35、CCC 约 0.29
- [x] 单独分析 severe 组系统性低估问题：severe 平均真实 BDI 约 34.14，平均预测约 17.64，平均残差约 -16.50
- [x] 单独分析 minimal 组系统性高估问题：minimal 平均真实 BDI 约 4.96，平均预测约 11.85，平均残差约 +6.89
- [x] 增加 Freeform/Northwind 同一 subject 预测一致性诊断，记录任务间预测差异和高差异 case
- [x] 在 Shortcut Audit 匹配修复后，重新判断 OpenFace pose/gaze/AU/quality 特征与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的相关性
- [x] 暂不将 OpenFace shortcut 特征直接加入训练输入；应先作为离线审计变量和 behavior-only baseline 对照使用
- [x] 将 severe 组高误差样本整理为 case study 清单，优先检查 `246_1`、`359_1`、`237_1`、`315_2`
- [x] 将 Freeform/Northwind 高差异样本整理为 case study 清单，优先检查 `237_1`、`247_1`、`247_3`、`224_1`、`212_1`
- [x] 建立 AU/pose/gaze/landmark-only behavior baseline 接口，与当前 RGB 模型在相同 split/seed/test checkpoint 下比较

## P0 剩余任务设计：从 shortcut 诊断转向可解释改进

当前 grouped-CV shortcut-only predictor 结果显示，OpenFace shortcut 特征不能单独接近 RGB/MTL-Lite 模型的测试表现；因此后续 P0 任务不应继续只围绕 backbone 微调或 in-sample shortcut 分数，而应优先解释模型为何出现预测范围压缩、severe 系统性低估、minimal 系统性高估，以及同一 subject 在 Freeform/Northwind 之间预测不一致。

- [x] P0-2：建立 high-error / task-inconsistency case study 清单。目标是把 severe 低估、minimal 高估、Freeform/Northwind 高差异和 low-error reference 分成可复查样本集合，为 attention、occlusion、keyframe、aligned face 逐案检查提供固定入口。
- [x] P0-2 输出设计：生成 `case_study_manifest.csv` 与 `case_study_manifest.md`，字段至少包括 `case_type`、`rank`、`video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`、`paired_task_pred_bdi`、`task_pred_diff`、`recommended_diagnostics`。
- [x] P0-2 判读重点：优先检查 `246_1`、`359_1`、`237_1`、`315_2` 等 severe 低估 subject，以及 `237_1`、`247_1`、`247_3`、`224_1`、`212_1` 等任务间高差异 subject；同时加入若干 low-error 样本作为对照。
- [x] P0-3：设计输入消融协议，但暂不直接改训练超参数。目标是判断模型是否依赖 RGB 纹理、边界伪影、身份线索、裁剪黑边或局部区域，而不是稳定面部行为动态。
- [x] P0-3 输入变体设计：`rgb` 作为当前 baseline；`grayscale` 弱化颜色线索；`blur` 弱化身份纹理；`center_mask` 保留面部中心；`boundary_erased` 弱化裁剪边界、头发、衣物残留和黑边。
- [x] P0-3 第一轮结果判读：`center_mask` 当前优于 `rgb`，而 `grayscale` 和 `blur` 变差；下一步应优先验证 OpenFace aligned face 的纯黑填充、黑色遮挡块和硬裁剪边界，而不是继续堆叠 late fusion 或新辅助任务。
- [x] P0-3 黑伪迹变体设计：`black_to_gray`、`black_to_mean`、`black_to_blur` 用于替换近黑像素；`soft_center_mask` 用于验证软边界是否优于硬 mask；`inner_crop_resize` 用于验证外围黑边是否为主要捷径。
- [ ] P0-3 服务器运行黑伪迹 ablation：保持与 `rgb`、`center_mask` 完全相同 split、seed、checkpoint 选择策略和指标。
- [ ] P0-3 汇总 `rgb`、`center_mask`、`boundary_erased` 与五个黑伪迹变体的整体 MAE/RMSE/Pearson/CCC、prediction mean/std、severity group error 和 task consistency。
- [ ] P0-3 运行黑伪迹审计，统计 `black_ratio`、`black_border_ratio`、`black_center_ratio`、`black_boundary_edge_ratio` 与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 的相关性。
- [ ] P0-3 对黑伪迹变体改善和恶化最明显的样本生成 case study 图组，重点检查麦克风黑块、脸部轮廓黑边、裁剪边界和模型关注区域。
- [ ] P0-3 后续补充：`landmark_heatmap` 应由 OpenFace landmark 坐标生成，归入 landmark/behavior baseline 路线，不能在只有 RGB 帧时伪造。
- [x] P0-3 评估约束：所有输入变体必须使用相同 split、seed、checkpoint 选择策略、训练入口和指标；优先记录 MAE、RMSE、Pearson、CCC、prediction mean/std、severe/minimal 分组误差和 Freeform/Northwind 一致性。
- [x] P0-4：设计 AU/pose/gaze/landmark-only behavior baseline 接口。目标是建立不依赖 RGB 纹理的行为表征对照，用来判断当前 RGB 模型是否真正捕捉到可泛化的行为动态。
- [x] P0-4 接口设计：新增 `src/datasets/openface_features.py`、`src/models/behavior_baseline.py`、`src/trainers/behavior_baseline_runner.py`、`scripts/train_behavior_baseline.py` 和 `configs/behavior_baseline.yaml`；输入包含 AU intensity/presence、pose、gaze、landmark、landmark temporal delta、confidence/success mask。
- [x] P0-4 判读方式：如果 behavior-only baseline 接近或超过 RGB/MTL-Lite，说明当前 RGB 输入中有大量可由结构化行为变量解释的有效信号；如果 behavior-only 显著弱于 RGB，但 RGB attribution 不集中在合理面部区域，则继续优先排查非行为捷径。
- [ ] 在服务器使用真实 OpenFace CSV 运行 behavior baseline debug smoke，并与 regression-only RGB baseline 使用相同 split/seed/metrics 对齐比较。

## Codex 任务队列

### Task 1

添加 `src/legacy/full_model/README.md`，完成旧模型归档说明。

### Task 2

新增 MTL-Lite 输出 dataclass 和 mask-aware pooling 工具。

### Task 3

新增 MTL-Lite 模型骨架。

### Task 4

新增 MTL-Lite forward/backward/config 测试。

### Task 5

新增 MTL-Lite baseline 配置和新训练入口。

### Task 6

新增 MTL-Lite 离线诊断与模型表征绘图系统。

### Task 7

整理 OpenFace 行为表征、相关论文和下一阶段实验路线，并将研究计划归档到文档。

### Task 8

构建 Shortcut Audit Framework 的最小可行实现：OpenFace quality summary、预测残差相关性、热力图和 markdown 报告。

## 2026-06-14 行为 baseline 后任务优先级重评估

最新 behavior-only baseline 训练结果显示：OpenFace 结构化特征路线可以在训练集上强拟合，但当前泛化不足。test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；best validation RMSE 约 `12.38`，但同一 epoch 的 train RMSE 只有约 `2.74`。因此，下一阶段任务重点应从“直接融合行为特征”调整为“先判断哪些 OpenFace 特征真正可泛化，哪些只是身份或静态几何捷径”。

### P0：必须立即处理

- [x] 为 behavior baseline 导出 val/test prediction CSV，并与 RGB/MTL-Lite prediction schema 对齐。
- [x] 在 behavior prediction 中记录 `video_id`、`subject_id`、`task_name`、`true_bdi`、`pred_bdi`、`residual`、`abs_error`、`severity_group`。
- [x] 为 behavior baseline 增加 `BEHAVIOR_FEATURES.FEATURE_SET` 命名特征组入口，支持后续以最小 override 运行特征组消融。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：quality-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：AU-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：pose+gaze-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：raw-landmark-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：landmark-delta-only。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：AU+landmark-delta。
- [ ] 为 behavior baseline 添加或运行 feature-group ablation：all-without-raw-landmarks。
- [ ] 对齐比较 RGB/MTL-Lite 与 behavior-only 的整体 MAE/RMSE/Pearson/CCC。
- [ ] 对齐比较 RGB/MTL-Lite 与 behavior-only 的 severe 低估、minimal 高估和 Freeform/Northwind task consistency。
- [ ] 分析 RGB 错而 behavior 对、behavior 错而 RGB 对、二者同时错误、二者同时正确的 case overlap。
- [x] 新增 RGB/MTL-Lite 与 behavior-only prediction CSV 离线比较入口，输出逐样本对照和整体/severity summary。
- [ ] 在训练日志或诊断报告中记录 OpenFace CSV 匹配数、可用字段、特征维度、缺失字段和训练集标准化统计来源。

### P1：强烈建议处理

- [ ] 在 feature-group ablation 后重新决定 behavior baseline 默认特征组，暂不默认相信 raw landmark 坐标。
- [ ] 尝试更小 behavior baseline 容量，例如减小 hidden dim、使用单向 GRU、增加 dropout 或 weight decay。
- [ ] 为 behavior baseline 引入更严格 early stopping，避免 train RMSE 继续下降但 val/test 不改善。
- [ ] 将 behavior prediction CSV 接入现有 regression diagnostics 和 case study manifest。
- [ ] 将 behavior feature ablation 结果整理为 `behavior_feature_ablation_results.csv`，便于论文表格化。

### P2：后续优化

- [ ] 在存在稳定可泛化 behavior 特征子集后，再设计 RGB + behavior late fusion。
- [ ] 在 behavior 特征子集稳定后，再设计 AU、landmark motion、pose/gaze 辅助任务的 MTL-Lite。
- [ ] 在辅助任务稳定后，再考虑 GradNorm、PCGrad、uncertainty weighting、LDS 或 `loss_dist` 消融。

## 2026-06-15 RGB 黑填充伪迹任务队列

当前用户更希望解释 RGB 输入模型过拟合原因，而不是继续堆叠多个任务。该判断是合理的：第一轮输入消融已经显示 `center_mask` 明显改善，说明输入侧非行为线索值得优先研究；如果不先定位 RGB 捷径，直接做 late fusion 或多任务可能只会把过拟合路径复杂化。

### P0：立即执行

- [x] 将黑填充/硬边界伪迹假设写入项目文档。
- [x] 接入黑伪迹输入变体配置和实现。
- [x] 接入黑伪迹离线审计脚本。
- [ ] 在服务器运行 `python -m pytest tests/test_input_variants.py`。
- [x] 使用相同命令模板运行五个黑伪迹 ablation：
  - `configs/input_ablation/black_to_gray.yaml`
  - `configs/input_ablation/black_to_mean.yaml`
  - `configs/input_ablation/black_to_blur.yaml`
  - `configs/input_ablation/soft_center_mask.yaml`
  - `configs/input_ablation/inner_crop_resize.yaml`
- [x] 对五组新实验全部运行 `scripts/diagnose_mtl_lite.py --enable-regression`，生成 `test_predictions.csv`、case study manifest 和回归诊断图。
- [x] 运行 `scripts/audit_black_artifacts.py`，至少先对原始 `rgb` 预测进行审计。

### P1：完成第一轮证据闭环

- [x] 建立人工分析 summary，汇总 `rgb`、`center_mask`、`boundary_erased` 和五个新变体。
- [x] 统计黑伪迹审计指标与误差之间的相关性，并完成判读。
- [x] 修正中心黑像素判读：中心近黑区域可能来自鼻孔、自然阴影、胡须、嘴角或麦克风遮挡，不能直接视为 OpenFace 伪迹。
- [x] 初步确认：黑边是泛化风险因子之一，但不是单独强解释变量。
- [ ] 将本轮 ablation 和黑伪迹审计整理成论文表格草稿。
- [ ] 对 severe 低估仍不改善的情况，继续保留 severity-aware loss/sampling、标签分布和 subject-level bias 作为后续独立问题。

### P1.5：下一轮精确边界黑区实验

- [x] 实现 `border_black_to_gray`：只替换与图像边界连通的近黑区域，不处理中心近黑像素。
- [x] 实现 `border_black_feather`：只对边界连通黑区做软过渡或 feather，降低硬边界突变。
- [x] 实现 `center_mask_black_to_gray`：在 `center_mask` 基础上处理残留边界连通黑区，验证二者是否互补。
- [x] 为上述三个变体新增 `configs/input_ablation/*.yaml`。
- [x] 为边界连通黑区 mask 增加单元测试，确保鼻孔、嘴角和麦克风等中心黑块不会被默认替换。
- [x] 本地完成 compile 验证：`src/datasets/input_variants.py` 与 `tests/test_input_variants.py` 语法检查通过。
- [ ] 在服务器运行 `python -m pytest tests/test_input_variants.py`。
- [ ] 在相同 split、seed、训练入口、checkpoint 策略下运行三组新 ablation。
- [ ] 将三组新结果与 `rgb`、`center_mask`、`black_to_gray`、`soft_center_mask` 统一比较。

### P1.6：case study 复核

- [ ] 高黑边高误差 case：`359_1`、`315_2`、`245_1`。
- [ ] 高黑边低误差 case：`247_3`。
- [ ] 低黑边高误差 case：`237_1`。
- [ ] `black_to_gray` 改善明显 case：`250_1`、`344_2`、`242_1`。
- [ ] `black_to_gray` 恶化明显 case：`206_2`、`226_2`、`210_2`。
- [ ] 对上述 case 生成 aligned frame montage、attention、spatial occlusion、keyframe 图组。
- [ ] 比较模型关注区域是否落在边界黑区、麦克风遮挡、鼻孔/嘴部自然暗区或真实面部行为区域。

### P2：暂缓

- [ ] RGB + behavior late fusion。
- [ ] AU / landmark / pose / gaze 辅助任务 MTL。
- [ ] 动态任务权重、PCGrad、GradNorm、LDS 或 `loss_dist`。
