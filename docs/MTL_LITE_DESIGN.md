# MTL-Lite 新架构设计

本文档描述项目从旧的大型端到端模型迁移到新主线 **MTL-Lite 轻量级多任务抑郁预测模型** 的代码结构、模块边界、接口定义和实施路线。

## 1. 架构决策

当前项目已经与最初的大型模型目标出现明显分化。为了避免新旧逻辑继续交织，后续采用硬边界策略：

```text
legacy/full_model：保存旧的大型端到端模型及其强相关模块
src/models：保存新主线模型和明确可复用的通用模型组件
src/diagnostics：保存独立诊断与可视化工具
```

核心原则：

- 旧模型整体迁移到 legacy 区域，作为可运行历史模型保留；
- 旧模型只保留说明文档和运行快照，不再投入额外修复或重构；
- 新模型不继承旧模型，不复用旧模型内部复杂训练路径；
- 可复用模块保留在主线位置；
- 旧模型专属模块放入 legacy；
- 诊断系统与模型主体解耦，尽量离线运行。

## 2. 目标模型主线

MTL-Lite 主流程：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

主任务：

- BDI 连续回归。

辅助任务：

- 有序抑郁严重程度分类。

论文主张：

使用有序严重程度辅助监督约束共享时序表征，在 AVEC2014 面部视频小样本场景中提升连续 BDI 预测的稳定性和可解释性。

### 2.1 OpenFace 行为表征扩展方向

当前输入帧已经经过 OpenFace 裁剪和对齐。后续模型设计需要承认 aligned face 中仍可能包含身份纹理、裁剪伪影、姿态残留、追踪质量和视频质量等非抑郁捷径。仅依赖 RGB backbone 可能不足以学习跨 subject 稳定的抑郁相关行为线索。

因此，MTL-Lite 的后续扩展方向从单纯 BDI ordinal 辅助监督，逐步转向 OpenFace 行为结构特征：

```text
aligned RGB frames
  + OpenFace landmarks / AU / pose / gaze / confidence
  -> RGB branch + behavior branch
  -> video-level fusion
  -> BDI regression + behavior-aware auxiliary tasks
```

建议优先实现和比较：

- landmark-only temporal baseline；
- AU / pose / gaze-only temporal baseline；
- RGB + behavior late fusion；
- AU intensity、AU presence、landmark motion、pose/gaze、expression distribution 等辅助任务。

该方向的研究依据和实验计划见 `docs/RESEARCH_NOTES.md`。

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

- `src/models/mtl_lite.py` 是新论文主模型；
- `src/models/outputs.py` 负责 dataclass 输出接口；
- `src/models/temporal/` 只放轻量、可复用的时序组件；
- `src/legacy/full_model/` 保存旧大模型及其专属依赖；
- `src/diagnostics/` 保存可视化与诊断工具，不作为模型结构贡献。

## 4. 保留在主线位置的通用模块

以下模块可被新旧模型或工具共同使用，建议保留在主线位置：

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
- 不再把旧 runner、旧脚本接回主线训练入口；
- 不在 legacy 中提交 `local_paths.yaml`、日志、权重、checkpoint 或实验结果；
- 只补充 `src/legacy/full_model/README.md`，说明旧模型边界、运行方式和维护策略。

旧模型如果需要运行，应从 legacy 快照自身路径和说明中运行，不作为新主线训练入口的一部分。

## 9. 配置接口设计

建议新增：

```yaml
MODEL:
  AUXILIARY_TASKS:
    ORDINAL_CLASSIFICATION: true
    CONTRASTIVE: false
  ENABLE_CGC: false
  ENABLE_ADAPTIVE_MASK: false
  ENABLE_PCGRAD: false
  ENABLE_UNCERTAINTY_WEIGHTING: false
```

建议新增 loss 配置：

```yaml
LOSSES:
  REGRESSION: "mse"
  ORDINAL_WEIGHT: 1.0
  CCC_WEIGHT: 0.0
  LDS_WEIGHTING: false
  DIST_WEIGHT: 0.0
```

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

模型主线需要轻量，但论文项目需要丰富的诊断与表征能力。

```text
模型主线：轻量、可解释、可复现、可消融
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
2. 说明 legacy 是旧大模型快照，不再作为新主线开发对象。
3. 说明旧模型如需运行，应使用 legacy 快照自身的脚本和配置。
4. 明确禁止提交 legacy 下的 `local_paths.yaml`、日志、权重和 checkpoint。
5. 不再投入时间修复 legacy 内部 import，除非用户明确要求复现旧模型结果。

### 阶段 2：新主线基础接口

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

目标：让新主线可以独立运行 regression-only 与 MTL-Lite baseline。

任务：

1. 新增 `scripts/train_mtl_lite.py`，作为 MTL-Lite 新主线训练入口。
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

### 阶段 7：消融实验

目标：在稳定 baseline 上逐项加入可选模块。

顺序：

1. OpenFace 质量与预测误差相关性；
2. 输入消融：RGB、grayscale、masked face、landmark heatmap、landmark/AU/pose only；
3. landmark-only temporal baseline；
4. AU / pose / gaze-only temporal baseline；
5. RGB + behavior late fusion；
6. 面部行为辅助任务 MTL；
7. MTL-Lite + CCC loss；
8. MTL-Lite + LDS；
9. MTL-Lite + `loss_dist`；
10. MTL-Lite + 动态任务权重或梯度冲突处理。

### 阶段 8：Shortcut Audit Framework

目标：在继续修改模型前，验证当前模型是否依赖非抑郁捷径。

任务：

1. 新增 OpenFace quality summary；
2. 合并 `predictions.csv`、OpenFace quality summary 和 split 信息；
3. 输出捷径变量与 BDI、预测值、残差、绝对误差的相关性；
4. 输出 shortcut-only BDI predictor baseline；
5. 输出 attention/occlusion 区域级统计；
6. 生成 `shortcut_audit_report.md`；
7. 所有诊断必须离线运行，不改变训练、验证或测试结果。

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
python scripts/diagnose_mtl_lite.py --run-dir /path/to/LOG_DIR/mtl_lite_csv/version_0 --ckpt best
```
