# MTL-Lite 模型基座设计

本文档描述项目从旧的大型端到端模型迁移到 **MTL-Lite 轻量级多任务抑郁预测模型基座** 的代码结构、模块边界、接口定义和实施路线。当前论文路线以 `docs/DOCS_GUIDE.md` 和 `docs/TODO.md` 为准；MTL-Lite 是训练和诊断基座，不单独覆盖当前 task-nuisance 路线。

## 1. 架构决策

当前项目已经与最初的大型模型目标出现明显分化。为了避免新旧逻辑继续交织，后续采用硬边界策略：

```text
legacy/full_model：保存旧的大型端到端模型及其强相关模块
src/models：保存 MTL-Lite 模型基座和明确可复用的通用模型组件
src/diagnostics：保存独立诊断与可视化工具
```

核心原则：

- 旧模型整体迁移到 legacy 区域，作为可运行历史模型保留；
- 旧模型只保留说明文档和运行快照，不再投入额外修复或重构；
- 新模型不继承旧模型，不复用旧模型内部复杂训练路径；
- 可复用模块保留在通用位置；
- 旧模型专属模块放入 legacy；
- 诊断系统与模型主体解耦，尽量离线运行。

## 2. 目标模型基座

MTL-Lite 主流程：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

主任务：

- BDI 连续回归。

辅助任务：

- 有序抑郁严重程度分类。

基座作用：

使用有序严重程度辅助监督约束共享时序表征，为 AVEC2014 面部视频小样本场景提供稳定、可消融的 BDI 预测基线。当前 task-nuisance 路线在该基座之上继续审计和改进表征。

### 2.1 OpenFace 行为表征对照（暂缓扩展）

当前输入帧已经经过 OpenFace 裁剪和对齐。后续模型设计需要承认 aligned face 中仍可能包含身份纹理、裁剪伪影、姿态残留、追踪质量和视频质量等非抑郁捷径。仅依赖 RGB backbone 可能不足以学习跨 subject 稳定的抑郁相关行为线索。

OpenFace 行为结构特征可以作为独立对照和证据来源，但不再作为当前 task-nuisance 路线的直接开发目标。下面是历史候选结构，不代表当前实现顺序：

```text
aligned RGB frames
  + OpenFace landmarks / AU / pose / gaze / confidence
  -> RGB branch + behavior branch
  -> video-level fusion
  -> BDI regression + behavior-aware auxiliary tasks
```

历史候选对照包括：

- landmark-only temporal baseline；
- AU / pose / gaze-only temporal baseline；
- behavior-only 与 RGB/MTL-Lite 的 prediction-level 对照；
- AU intensity、AU presence、landmark motion、pose/gaze、expression distribution 等辅助任务仅在行为特征子集稳定后再考虑。

该方向的历史研究依据见 `docs/RESEARCH_NOTES.md`。当前执行顺序仍以 `docs/TODO.md` 的 Stage A/B/C/D 为准。

非抑郁捷径验证的具体实施方案见 `docs/SHORTCUT_AUDIT_DESIGN.md`。该框架应作为模型改动前的离线诊断层，优先验证 OpenFace 质量、姿态、gaze、裁剪伪影和预测误差之间的关系。

## 3. 推荐目录结构

目标结构：

```text
src/
  config.py
  datasets/
    dataset.py
  metrics/
    metrics.py
  models/
    __init__.py
    backbone_factory.py
    task_heads.py
    outputs.py
    mtl_lite.py
    temporal/
      __init__.py
      encoders.py
      pooling.py
  diagnostics/
    __init__.py
    regression.py
    embeddings.py
    temporal.py
    attribution.py
    reports.py
  legacy/
    full_model/
      README.md
      src/
        models/
          end_to_end.py
          mtl_blocks.py
          iresnet.py
        utils/
          adaptive_mask.py
          decomposition.py
          label_distribution.py
          pcgrad.py
          visualize.py
        losses/
          losses.py
        trainers/
          end_to_end_runner.py
      scripts/
      configs/
      tests/
```

说明：

- `src/models/mtl_lite.py` 是当前训练和诊断的模型基座；
- `src/models/outputs.py` 负责 dataclass 输出接口；
- `src/models/temporal/` 只放轻量、可复用的时序组件；
- `src/legacy/full_model/` 保存旧大模型及其专属依赖；
- `src/diagnostics/` 保存可视化与诊断工具，不作为模型结构贡献。

## 4. 保留在通用位置的模块

以下模块可被新旧模型或工具共同使用，建议保留在通用位置：

- `src/config.py`
- `src/datasets/dataset.py`
- `src/metrics/metrics.py`
- `src/models/backbone_factory.py`
- `src/models/task_heads.py`

保留理由：

- 它们属于项目基础设施或通用建模组件；
- 与旧模型专属算法没有强绑定；
- MTL-Lite、baseline、诊断脚本都可能继续使用。

## 5. 放入 legacy 的旧模型模块

以下模块应归入 `src/legacy/full_model/`：

- 旧版 `EndToEndDepressionModel`
- CGC / expert routing
- contrastive learning head/loss
- adaptive mask
- PCGrad
- uncertainty weighting
- LDS label weighting
- `loss_dist` 训练路径
- segmented weight save/load 逻辑
- 旧版诊断 hooks
- 旧版 runner、脚本、配置和测试快照

legacy 的目标不是删除旧能力，而是保存一个可追溯、可独立运行、边界清晰的历史模型快照。由于旧模型复制到新路径后已经可以按旧方式运行，后续不再花费主要精力修复或重构旧模型内部代码。legacy 只需要 README 说明边界、运行方式和不再扩展的维护策略。

## 6. 输出接口设计

建议新增 `src/models/outputs.py`。

```python
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MTLLiteOutput:
    bdi_pred: torch.Tensor
    ordinal_logits: Optional[torch.Tensor] = None
    shared_features: Optional[torch.Tensor] = None


@dataclass
class MTLLiteLosses:
    total: torch.Tensor
    regression: torch.Tensor
    ordinal: Optional[torch.Tensor] = None
    ccc: Optional[torch.Tensor] = None
```

约定：

- `bdi_pred` 使用 normalized BDI 尺度；
- 指标计算前通过 `prediction_for_metrics()` 恢复到真实 BDI 尺度；
- `ordinal_logits` 对应 CORAL 或其他有序分类形式；
- `shared_features` 只在诊断或可视化时返回；
- loss 输出必须区分总 loss 和子 loss。

## 7. MTL-Lite 模型接口

建议新增 `src/models/mtl_lite.py`。

```python
class MTLLiteDepressionModel(pl.LightningModule):
    def __init__(self, configs):
        ...

    def extract_frame_features(self, video_tensor, mask):
        ...

    def encode_temporal_features(self, frame_features, mask):
        ...

    def pool_video_features(self, temporal_features, mask):
        ...

    def forward(self, video_tensor, mask, return_features=False) -> MTLLiteOutput:
        ...

    def prepare_labels(self, labels):
        ...

    def compute_losses(self, outputs: MTLLiteOutput, labels) -> MTLLiteLosses:
        ...

    def prediction_for_metrics(self, bdi_preds):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...
```

接口职责：

- `extract_frame_features()`：只负责 backbone 与 padding mask；
- `encode_temporal_features()`：只负责时序编码；
- `pool_video_features()`：只负责视频级 mask-aware pooling；
- `forward()`：只返回模型输出；
- `compute_losses()`：只负责 loss 计算；
- `prediction_for_metrics()`：只负责尺度恢复；
- Lightning step：只负责组织 batch、调用 forward、记录 loss/metric。

## 8. legacy 维护策略

旧模型后续按以下原则维护：

- 只作为历史快照和回退参考；
- 不再承载 MTL-Lite 新逻辑；
- 不主动修复旧模型内部 import，除非阻塞历史结果复现；
- 不再把旧 runner、旧脚本接回 MTL-Lite 基座训练入口；
- 不在 legacy 中提交 `local_paths.yaml`、日志、权重、checkpoint 或实验结果；
- 只补充 `src/legacy/full_model/README.md`，说明旧模型边界、运行方式和维护策略。

旧模型如果需要运行，应从 legacy 快照自身路径和说明中运行，不作为 MTL-Lite 基座训练入口的一部分。

## 9. 配置接口设计

MTL-Lite 基线只保留当前实际使用的键：

```yaml
MODEL:
  AUXILIARY_TASKS:
    ORDINAL_CLASSIFICATION: True

LOSSES:
  ORDINAL_WEIGHT: 1.0
  CCC_WEIGHT: 0.0

DATASET:
  INPUT_VARIANT: "rgb"
```

历史遗留键（如 `MODEL.ENABLE_CGC`、`MODEL.ENABLE_ADAPTIVE_MASK`、`LOSSES.REGRESSION`、`LOSSES.LDS_WEIGHTING`、`LOSSES.DIST_WEIGHT`、`VISUALIZATION` 等）已从 `configs/avec2014_base.yaml` 移除，完整旧配置保留在 `configs/pre/default_config.yaml`。

当前 RGB dataset 支持 `rgb`、`grayscale`、`blur`、`center_mask`、`central_face_mask`、`boundary_erased`、`black_to_gray`、`black_to_mean`、`black_to_blur`、`soft_center_mask`、`inner_crop_resize`。`landmark_heatmap` 需要真实 OpenFace landmark 坐标，应在后续 behavior baseline 或 OpenFace landmark dataset 中实现，不应由 RGB 帧伪造。

2026-06-15 之后，输入消融的历史审计目标从泛泛验证“背景/纹理捷径”收窄到 OpenFace aligned face 的黑填充和硬边界伪迹。`center_mask` 优于 `rgb`，但它可能同时改变了面部区域和黑边伪迹，因此当时需要通过 `black_to_*`、`soft_center_mask` 和 `inner_crop_resize` 继续拆解原因。当前这些结果作为 Stage A 证据层保留，RGB + behavior late fusion 和新的行为辅助任务仍不作为当前模型路线。

2026-07-03 对真实输入帧的审查显示，历史 `center_mask` 只保留鼻口附近小区域，不能解释为完整中心脸或行为区域。`central_face_mask` 是新增的语义校准消融，用于覆盖眼、鼻、嘴和主要脸颊，并验证历史 `center_mask` 结论是否来自极强遮挡。

黑伪迹审计后的历史实现收窄原则：中心近黑像素可能是鼻孔、嘴角阴影、胡须、麦克风或真实遮挡，不应默认替换。当时输入变体实现了 `border_black_to_gray`、`border_black_feather` 和 `center_mask_black_to_gray`，只处理中与图像边界连通的近黑区域，并保留中心近黑语义。

为了避免破坏现有配置，新模型实现应对缺失字段提供默认值。

backbone 可训练范围由以下配置控制：

```yaml
EXTRACT_FEATURE:
  FREEZE_BACKBONE: true
  FINETUNE_LAST_N_BLOCKS: 1
```

约定：

- `FREEZE_BACKBONE: false`：保持 backbone 全量可训练；
- `FREEZE_BACKBONE: true` 且 `FINETUNE_LAST_N_BLOCKS: 0`：全冻结 backbone；
- `FREEZE_BACKBONE: true` 且 `FINETUNE_LAST_N_BLOCKS > 0`：冻结 backbone 大部分参数，只解冻最后若干 transformer blocks 和 norm；
- 对没有 `.blocks` 的 backbone，`FINETUNE_LAST_N_BLOCKS` 不强行猜测 CNN 层级，保持全冻结并打印提示。

## 10. 诊断与可视化系统

MTL-Lite 基座需要轻量，但论文项目需要丰富的诊断与表征能力。

```text
MTL-Lite 基座：轻量、可解释、可复现、可消融
诊断系统：丰富、模块化、离线运行、支持论文分析
```

推荐后续结构：

```text
src/diagnostics/
  io.py
  training_curves.py
  regression.py
  embeddings.py
  correlation.py
  occlusion.py
  keyframes.py
  model_attention.py
  reports.py
```

建议支持：

- loss/metric 曲线；
- prediction-target scatter；
- residual histogram；
- BDI 区间误差；
- severity group 误差；
- high-error / low-error subject ranking；
- embedding、t-SNE、UMAP；
- metrics / predictions 相关系数热力图；
- 遮掩影响热力图；
- 关键帧重要性热力图；
- 模型自身关注区域热力图；
- temporal weights / gating；
- Grad-CAM、attention CAM、occlusion sensitivity；
- subject-level case study 图组。

诊断逻辑不得改变训练、验证或测试结果。

## 11. 实施路线

### 阶段 1：legacy 归档说明

目标：确认旧模型已经作为 legacy 快照保存，并用 README 划清维护边界。不再对旧模型进行额外修复或结构重构。

任务：

1. 添加 `src/legacy/full_model/README.md`。
2. 说明 legacy 是旧大模型快照，不再作为当前开发对象。
3. 说明旧模型如需运行，应使用 legacy 快照自身的脚本和配置。
4. 明确禁止提交 legacy 下的 `local_paths.yaml`、日志、权重和 checkpoint。
5. 不再投入时间修复 legacy 内部 import，除非用户明确要求复现旧模型结果。

### 阶段 2：MTL-Lite 基座基础接口

目标：为 MTL-Lite 建立干净的输出、时序池化和模型接口。

任务：

1. 新增 `src/models/outputs.py`。
2. 定义 `MTLLiteOutput`。
3. 定义 `MTLLiteLosses`。
4. 新增 `src/models/temporal/__init__.py`。
5. 新增 `src/models/temporal/pooling.py`。
6. 实现 `masked_mean_pool`。

### 阶段 3：MTL-Lite 骨架

目标：建立新模型，不替换旧模型。

任务：

1. 新增 `src/models/mtl_lite.py`。
2. 实现 `MTLLiteDepressionModel` 骨架。
3. 只依赖通用 backbone、task heads、metrics 和 temporal utilities。
4. 不继承 legacy full model。
5. 不引入 contrastive、CGC、adaptive mask、PCGrad、LDS 或 `loss_dist`。

### 阶段 4：新训练入口与配置

目标：让 MTL-Lite 基座可以独立运行 regression-only 与 MTL-Lite baseline。

任务：

1. 新增 `scripts/train_mtl_lite.py`，作为 MTL-Lite 基座训练入口。
2. 新增 `src/trainers/mtl_lite_runner.py`，封装 MTL-Lite Lightning trainer、logger、checkpoint 和 test 流程。
3. 新增 `configs/regression_only_baseline.yaml`，用于 BDI 回归单任务 baseline。
4. 新增 `configs/mtl_lite_baseline.yaml`，用于 BDI 回归 + 有序严重程度分类 baseline。
5. 新增 `configs/mtl_lite_debug_smoke.yaml`，用于服务器快速 smoke。
6. `AVECDataset` 通过 `DATASET.RETURN_MULTI_VIEW_TRAIN` 控制训练集是否返回旧多视图结构；MTL-Lite 配置应设置为 `False`。
7. 在服务器运行 MTL-Lite smoke。

### 阶段 5：MTL-Lite 测试

目标：在接入训练入口前验证新模型基本行为。

任务：

1. 新增 `tests/test_mtl_lite_forward.py`。
2. 新增 `tests/test_mtl_lite_loss_backward.py`。
3. 新增 `tests/test_mtl_lite_config.py`。
4. 使用 dummy backbone，不下载权重，不读取真实数据。
5. 检查 forward shape、loss finite、regression head 梯度非零、loss/metric 尺度一致。

### 阶段 6：诊断系统独立化

目标：将可视化和诊断从模型中分离。

任务：

1. 梳理 `src/utils/visualize.py`。
2. 新增 `src/diagnostics/`。
3. 支持离线读取 `metrics.csv`、`predictions.csv`、features 和 temporal weights。
4. 支持训练曲线、prediction-target scatter、residual histogram、BDI 区间误差、severity group 误差、high/low error ranking。
5. 支持 embedding PCA/t-SNE/UMAP。
6. 支持 metrics / predictions 相关系数热力图。
7. 支持遮掩影响热力图和关键帧重要性热力图。
8. 支持模型关注区域热力图：Grad-CAM 可用时优先，否则回退到 input-gradient attention。
9. 保留旧入口兼容。

### 阶段 7：历史消融实验记录

目标：记录 MTL-Lite 基座阶段曾规划或完成的输入、行为和损失消融。当前 task-nuisance 路线不再按该列表顺序扩展模型；新的执行顺序见 `docs/TODO.md` 的 Stage A/B/C/D。

顺序：

1. OpenFace 质量与预测误差相关性；
2. 输入消融：RGB、grayscale、blur、center_mask、boundary_erased、black_to_gray、black_to_mean、black_to_blur、soft_center_mask、inner_crop_resize、landmark heatmap、landmark/AU/pose only；
3. landmark-only temporal baseline；
4. AU / pose / gaze-only temporal baseline；
5. RGB + behavior late fusion（暂缓）；
6. 面部行为辅助任务 MTL（暂缓）；
7. MTL-Lite + CCC loss；
8. MTL-Lite + LDS；
9. MTL-Lite + `loss_dist`；
10. MTL-Lite + 动态任务权重或梯度冲突处理。

### 阶段 7.1：OpenFace behavior-only baseline

当前已新增独立 behavior-only baseline 路线：

```text
OpenFace CSV
  -> AU / pose / gaze / landmark / confidence / success sequence
  -> temporal delta / optional acceleration
  -> GRU temporal encoder
  -> BDI regression head
```

实现位置：

```text
src/datasets/openface_features.py
src/models/behavior_baseline.py
src/trainers/behavior_baseline_runner.py
scripts/train_behavior_baseline.py
configs/behavior_baseline.yaml
```

该路线用于判断结构化行为变量是否可以解释当前 RGB 模型的有效信号。它不依赖 RGB visual backbone，不应并入 `MTLLiteDepressionModel`。在当前路线下，它作为独立对照和 Stage A 证据来源；late fusion 继续暂缓，除非后续证据证明行为特征子集稳定且 task-nuisance 路线仍需要补充。

### 阶段 8：Shortcut Audit Framework

目标：在继续修改模型前，验证当前模型是否依赖非抑郁捷径。

任务：

1. 新增 OpenFace quality summary；
2. 合并 `predictions.csv`、OpenFace quality summary 和 split 信息；
3. 输出捷径变量与 BDI、预测值、残差、绝对误差的相关性；
4. 输出 shortcut-only BDI predictor baseline；
5. 输出 attention/occlusion 区域级统计；
6. 生成 `shortcut_audit_report.md`；
7. 新增黑填充/硬边界伪迹审计，输出 `black_artifact_audit_report.md`；
8. 所有诊断必须离线运行，不改变训练、验证或测试结果。

## 12. 验证命令

每次代码修改后建议运行：

```bash
python -m compileall src scripts tests
python -m pytest tests/test_model_forward.py tests/test_loss_backward.py tests/test_loss_metric_consistency.py
```

legacy 归档说明完成后：

```bash
git diff -- src/legacy/full_model/README.md
python -c "from src.models.backbone_factory import build_feature_backbone; print('backbone import ok')"
python -c "from src.models.task_heads import build_regression_task_head; print('task heads import ok')"
```

新增 MTL-Lite 后：

```bash
python -c "from src.models.mtl_lite import MTLLiteDepressionModel; print('mtl lite import ok')"
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
python scripts/diagnose_mtl_lite.py --run-dir <LOG_DIR>/default/mtl_lite/version_0 --ckpt best
```

## 13. Stage B 设计规格（B0 规格冻结）

本节是 Stage B 的最小设计规格，对应 `docs/TODO.md` 的 **B0-spec-first** 任务包与 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的 Stage B 实施路线。B0 阶段只冻结设计、配置草案与测试计划；**不修改 `src/models/mtl_lite.py`**，模型代码在 B1/B2 才落地。

Stage A 已收口（A1 层级身份探针、A2 残差-身份耦合、A4 severity 压缩同时成立，A3 matched-only 仅作 probe/case/group-wise 评估），因此 Stage B 的工程目标是把两类已证实的问题拆成最小可控干预：身份捷径用 GRL/subject attacker 检验，severity 压缩用训练侧重加权检验。代码实现不得改变默认 RGB baseline 行为；所有新能力都必须通过配置显式打开。

### 13.1 范围与边界

**特征入口固定**

- Stage B 的 `z_dep` 就是当前 MTL-Lite 的 `shared_features`，即 `pool_video_features(...)` 输出、进入 `reg_task_head` 与 `ordinal_task_head` 之前的共享表征（`src/models/mtl_lite.py:387`）。
- Stage B **不新增** `TaskNuisanceBlock`，不划分 `z_nuisance`，不引入显式 `z_art / z_ctx / z_pose / z_quality`。这些属于 Stage C 范围，且仅在 B5 判定 B 不足以改善风险后才进入。

**配置开关**

- 只允许新增两类开关：`identity_adversarial`（B1）与 `severity_balanced_regression`（B2），**默认均为关闭**。
- 开关关闭时，`forward` / `compute_losses` / 数值结果必须与当前 E0 baseline 逐位等价，既有测试不应有任何变化。

**实验同质性**

- E0/E1/E2/E3 使用同一 split、seed、backbone、输入变体、optimizer、precision、checkpoint 策略和 metric 输出路径。
- runner 的 `ModelCheckpoint(monitor="val_RMSE_epoch", mode="min")` 不得改动，确保四组按同一口径选最优 checkpoint。

**禁止边界（Stage B 不得触碰）**

- `TaskNuisanceBlock`、`z_nuisance`、显式 `z_art / z_ctx / z_pose / z_quality` latent；
- 新增 RGB 输入滤镜、dynamic branch、late fusion；
- 改变默认 baseline 行为（开关默认关闭、配置缺省时数值等价）；
- 从 val/test 扩展 identity 类别表或用 val/test 分布调 severity 权重。

### 13.2 接口变更草案

B0 只规定接口形态，B1/B2 落地时按下述签名实现。所有新增字段均 Optional 且默认 `None`，保证旧调用路径不受影响。

**输出与损失结构扩展**（`src/models/outputs.py`）：

```python
@dataclass
class MTLLiteOutput:
    bdi_pred: torch.Tensor
    ordinal_logits: Optional[torch.Tensor] = None
    shared_features: Optional[torch.Tensor] = None
    layer_features: Optional[Dict[str, torch.Tensor]] = None
    # B1: subject-id head 输出，GRL 之后的 logits；开关关闭时为 None。
    identity_logits: Optional[torch.Tensor] = None


@dataclass
class MTLLiteLosses:
    total: torch.Tensor
    regression: Optional[torch.Tensor] = None
    ordinal: Optional[torch.Tensor] = None
    ccc: Optional[torch.Tensor] = None
    # B1: subject-id CE loss，仅 train stage 且 batch 内 subject 全部可映射时非 None。
    identity: Optional[torch.Tensor] = None
```

**前向分支**（`MTLLiteDepressionModel.forward`）：`shared_features` 已在现有 forward 中计算，B1 只在 `identity_adversarial` 打开时追加一条分支，不重写既有路径。

```text
shared_features -> reg_task_head            -> bdi_pred
shared_features -> ordinal_task_head        -> ordinal_logits   (if ordinal)
shared_features -> GradReverse(lambda_id)   -> subject_id_head  -> identity_logits  (if identity_adversarial)
```

**损失签名**：`compute_losses` 需要知道当前 stage 以 gating identity loss。最小改动是把 stage 透传进去，`_shared_step` 已持有 `stage`，调用点改为 `compute_losses(outputs, labels, stage=stage)`。

```python
def compute_losses(self, outputs, labels, stage="train"):
    _, true_bdi_norm, ordinal_levels = self.prepare_labels(labels)
    # B2: severity-bin reweighting（开关关闭时退化为标准 MSE）。
    loss_reg = self._regression_mse(outputs.bdi_pred, true_bdi_norm, labels)
    loss_ccc = concordance_ccc_loss(outputs.bdi_pred, true_bdi_norm)
    loss_ord = None
    if outputs.ordinal_logits is not None and self.ordinal_weight > 0.0:
        loss_ord = coral_loss(outputs.ordinal_logits, ordinal_levels)

    # B1: identity loss 仅在 train stage、且 batch 内 subject 全部命中 train 类别表时启用。
    loss_id = None
    if outputs.identity_logits is not None and stage == "train":
        subject_index = self._map_subjects_to_index(labels["subject_id"])
        if subject_index is not None:
            loss_id = F.cross_entropy(outputs.identity_logits, subject_index)

    total = loss_reg + self.ccc_loss_weight * loss_ccc
    if loss_ord is not None:
        total = total + self.ordinal_weight * loss_ord
    if loss_id is not None:
        total = total + loss_id   # lambda 已在 GRL 中体现，这里不再加权
    return MTLLiteLosses(total=total, regression=loss_reg, ordinal=loss_ord,
                         ccc=loss_ccc, identity=loss_id)
```

**subject 类别表构建**：`AVECDataModule` 已在 runner 内构建；B1 在 `run_mtl_lite` 中（或 data module setup 后）从 **train split** 统计 subject，构建 `subject_id_to_index` 并注入模型（如 `model.set_subject_index(...)`）。val/test 的 unseen subject 不进入类别表，`_map_subjects_to_index` 对未命中返回 `None`，该 batch 不产生 identity loss 但不报错。`labels["subject_id"]` 数据集已提供（`src/datasets/dataset.py:207`），无需改数据层。

### 13.3 配置草案

沿用现有 YAML 嵌套段风格（参考 `MODEL.AUXILIARY_TASKS.ORDINAL_CLASSIFICATION` 的读取方式）。默认值在模型 `_get_nested_config_value` 中保持向后兼容。

```yaml
MODEL:
  IDENTITY_ADVERSARIAL:
    ENABLE: False            # B1 总开关，默认关闭
    LAMBDA_ID: 0.05          # GRL 系数，sweep: 0.02 / 0.05 / 0.10 / 0.20
    NUM_SUBJECT_CLASSES: 0   # 由 runner 从 train split 注入，配置中只占位
  SEVERITY_BALANCED_REGRESSION:
    ENABLE: False            # B2 总开关，默认关闭
    POWER: 0.5               # 先 0.5，再比较 1.0
    MIN_WEIGHT: 0.5          # 截断下界
    MAX_WEIGHT: 4.0          # 截断上界
    EDGES: [13, 19, 28]      # 复用 src/diagnostics/io.py:severity_group 边界
```

**固定实验矩阵配置文件**（`configs/stage_b/`，B1/B2 落地时创建）：

```text
configs/stage_b/e0_rgb_mtl_lite.yaml                       # 两开关均 False（或省略），即现 baseline
configs/stage_b/e1_identity_adversarial.yaml               # IDENTITY_ADVERSARIAL.ENABLE True
configs/stage_b/e2_severity_balanced.yaml                  # SEVERITY_BALANCED_REGRESSION.ENABLE True
configs/stage_b/e3_identity_adversarial_severity_balanced.yaml  # 两开关均 True
```

E0 若能由现有 baseline 配置复现，可只记录 resolved config 与运行命令，不强制新增重复文件。`lambda_id` 与 `power` 的 sweep 通过同一份配置 + CLI override 产生多 run，不复制多份近似配置。

### 13.4 B1 Identity-Adversarial 接口规格

- 数据流：`shared_features -> GRL(lambda_id) -> subject_id_head -> identity_logits`。
- subject 类别只来自 train split subject；val/test unseen subject 不进类别表、不参与 identity loss。
- identity loss 仅在 `stage == "train"` 且 batch 内 subject 全部可映射时启用；val/test 的 BDI loss 口径与 E0 完全可比。
- GRL 初始 sweep 固定 `lambda_id = 0.02, 0.05, 0.10, 0.20`，优先选“不损伤 BDI utility 的最低有效强度”。
- 诊断必须同时报告训练中的 subject-head accuracy 与离线 A1/A2 identity risk；**不能只用训练 adversarial loss 证明去身份成功**。

**通过条件**

- `layer_shared` 或最终表征的 same-subject top1/top5、paired rank、subject attacker risk 至少有明确下降趋势。
- MAE/RMSE/CCC 不出现明显崩坏，severity bias 和 task consistency 不比 E0 明显恶化。
- 若 identity risk 下降但 CCC、severity agreement 或 Freeform/Northwind consistency 明显下降，**不视为成功**。

### 13.5 B2 Severity-Balanced 接口规格

- 损失形式：`loss_reg = mean( weight(severity_bin(y_i)) * MSE(pred_i, y_i_norm) )`，按样本加权后求 batch 均值。
- 第一版**只对 regression MSE 做 severity-bin reweighting**；CCC loss 与 ordinal auxiliary loss 暂不加权，避免 batch 级 CCC/ordinal 解释混在一起。
- severity bin 固定为 `minimal / mild / moderate / severe`，边界复用 `src/diagnostics/io.py:42` 的 `severity_group`（`≤13 / ≤19 / ≤28 / >28`），保证与 A4/severity 诊断同口径。
- 权重公式：`weight = (total / (num_bins * count_bin)) ** power`，只从 **train split** 标签计数计算，先跑 `power=0.5`，再比较 `power=1.0`。
- 权重需 mean-normalize，并设置 `MIN_WEIGHT / MAX_WEIGHT` 截断，避免小样本 severe bin 让梯度失控。
- 训练日志必须记录每个 bin 的 count、raw weight、clipped weight、final normalized weight。
- 第一版不引入 balanced sampler、Huber/MAE/CCC 混合 sweep 或 ordinal 改造；这些只作为 Stage B 失败后的局部扩展。

```python
# 仅由 train split 计数计算
counts = {bin: count_in_train(bin) for bin in ("minimal","mild","moderate","severe")}
total, num_bins = sum(counts.values()), 4
raw   = {b: (total / (num_bins * counts[b])) ** power for b in counts}
clipped= {b: min(max(raw[b], MIN_WEIGHT), MAX_WEIGHT) for b in counts}
norm  = {b: clipped[b] / mean(clipped.values()) for b in counts}   # mean-normalize
per_sample_w = norm[severity_bin(y_i)]                             # y_i 为原始 BDI 分
loss_reg = (per_sample_w * mse_per_sample).mean()
```

**通过条件**

- minimal 高估和 severe 低估相对 E0 有实质改善，且 `pred_std / true_std` 更接近真实分布。
- CCC 不明显下降，identity risk 不升高，task consistency 不恶化。
- 若只是整体均值上移导致 severe bias 变小、但 minimal bias 或 CCC 明显恶化，**不视为成功**。

### 13.6 B3 固定实验矩阵

```text
E0  RGB MTL-Lite baseline
E1  identity-adversarial MTL
E2  severity-balanced regression
E3  identity-adversarial MTL + severity-balanced regression
```

运行约束：四组必须使用同一 split、seed、input variant、optimizer、precision、checkpoint 策略与输出目录结构；`lambda_id` 与 `power` 的 sweep 在每组内部展开为多 run，但不得跨组改动其它变量。

### 13.7 B4 诊断与汇总

每个实验完成后至少输出五类诊断，统一以 E0 为基准横向对照（不写孤立结论）：

1. **prediction summary**：MAE/RMSE/Pearson/CCC、pred mean/std、train-val gap；
2. **severity summary**：minimal/mild/moderate/severe 的 count、MAE、bias、abs error；
3. **identity summary**：A1 layer/shared retrieval、A2 residual-identity coupling、subject attacker accuracy；
4. **task consistency**：同 subject Freeform/Northwind prediction diff 与 residual diff；
5. **artifact-risk group**：基于 matched-only A3 变量做分组评估，**不把 A3 变量变成训练监督**。

### 13.8 B5 阶段判定

- **E1 有效**：identity risk 下降且 BDI/severity/task consistency 未明显恶化 → 保留 identity-adversarial 作为 Stage C 必要对照，后续在 `z_dep` 上继续报告 identity leakage。
- **E2 有效**：minimal/severe bias 与 compression 改善，CCC 与 identity risk 未明显恶化 → 保留 severity-balanced regression 作为 Stage C/D 支线，但仍需检查它是否提高 identity risk。
- **E3 明显优于 E1/E2**：说明 identity shortcut 与 severity compression 存在互补干预价值 → Stage C 必须以 E3 作为强 baseline。
- **均不能在可接受代价内改善风险**：才进入 Stage C 的粗粒度 `z_dep / z_nuisance` 解耦（C0-spec-before-code）。

### 13.9 测试计划（B1/B2 落地时必备，B0 仅记录）

**B1 必备测试**

- import check：`from src.models.mtl_lite import MTLLiteDepressionModel`；
- config loading check：缺省配置下 `identity_adversarial` 为 False，`identity_logits` 与 `loss.identity` 均为 `None`；
- dummy one-batch forward/loss：输出形状正确、loss finite、reg head 梯度非零；
- **GRL 开关不影响默认 baseline**：开关关闭时 `total` 与现有 E0 逐位等价；
- **val/test unknown subject 不触发 loss**：含 unseen subject 的 batch 下 `loss.identity` 为 `None` 且不报错；
- `subject_id_to_index` 仅由 train split 构建，unseen subject 映射返回 `None`。

**B2 必备测试**

- 权重计算：由 train 计数生成、mean-normalize、clip 到 `[MIN_WEIGHT, MAX_WEIGHT]`；
- loss 缩放：severity-bin reweighting 后 `loss_reg` 相对 plain MSE 的尺度变化符合权重表；
- **默认关闭不改变 baseline**：开关关闭时 `loss.regression` 与 plain MSE 逐位等价；
- **只改 regression MSE 不改 CCC/ordinal**：开关打开时 CCC 与 ordinal loss 的逐样本值与关闭时一致，仅 regression 项被重加权。

### 13.10 B0 阶段交付清单

- [x] 本节设计规格（范围/边界、接口草案、配置草案、B1/B2 规格、B3 矩阵、B4 诊断、B5 判定、测试计划）；
- [x] `configs/stage_b/` 四份配置草案（E0/E1/E2/E3）+ README（四组全部落地）；
- [x] B1 测试用例（`tests/test_mtl_lite_identity_adversarial.py` + `tests/test_gradient_reversal.py`）；
- [x] B2 测试用例（`tests/test_mtl_lite_severity_balanced.py`，8 项；权重公式/clip/normalize、bin 边界与 `io.severity_group` 同口径、加权 MSE=逐样本加权、默认关闭=plain MSE、只改 regression 不改 CCC/ordinal、degenerate 回退）；
- [x] B1 代码：`src/models/outputs.py`（`identity_logits` / `identity`）、`src/models/gradient_reversal.py`（GRL）、`src/models/mtl_lite.py`（`set_subject_index` / `_map_subjects_to_index` / forward 分支 / `compute_losses(..., stage=)` gating / `_shared_step` 透传 stage）、`src/trainers/mtl_lite_runner.py`（`build_train_subject_index` + 注入）；
- [x] B2 代码：`src/models/mtl_lite.py`（`set_severity_bin_counts` / `_severity_bin_index` / `_severity_weighted_mse` / `compute_losses` regression 项替换）、`src/trainers/mtl_lite_runner.py`（`build_train_severity_bin_counts` + 注入，edges 取自模型）。两个开关默认关闭时与 E0 逐位等价（27 项测试覆盖）。

B0 完成后按 `docs/TODO.md` 的 **B1-code-minimal** → **B2-severity-balanced-minimal** → **B3-fixed-experiments** → **B4-stage-b-report** → **B5-stage-b-gate** 顺序推进。**B1-code-minimal 与 B2-severity-balanced-minimal 均已完成**，下一步进入 **B3-fixed-experiments**（在服务器上按统一 split/seed/backbone 跑完 E0–E3，含 `lambda_id` 与 `POWER` 的 sweep）。
