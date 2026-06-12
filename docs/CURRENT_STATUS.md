# CURRENT_STATUS.md

## 状态日期

2026-06-12

## 当前项目状态

项目已经通过服务器 debug smoke，可以完整运行旧版端到端训练流程。当前工作重点已经从“修复可运行性”转向“重构架构边界”，为 MTL-Lite 轻量级多任务抑郁预测模型建立干净主线。

## 当前架构方向

项目采用以下新架构：

```text
src/legacy/full_model/  # 旧大模型整体归档
src/models/             # MTL-Lite 新模型与通用模型模块
src/diagnostics/        # 独立诊断与可视化系统
```

关键决策：

- 旧大模型整体进入 legacy；
- 旧模型只作为历史快照保留，暂时不再投入额外修复或重构；
- 新模型不继承旧模型；
- 通用模块保留在主线位置；
- 新主线训练入口后续直接围绕 MTL-Lite 设计；
- 诊断与可视化系统从模型训练逻辑中解耦。

## 当前研究主线

论文主线是 **MTL-Lite BDI 预测模型**：

```text
人脸视频帧 -> 视觉 backbone -> 时序编码器 -> 共享视频表征 -> BDI 回归头 + 有序严重程度分类头
```

主任务：

- BDI 连续回归。

辅助任务：

- 有序抑郁严重程度分类。

非主线模块：

- contrastive learning
- adaptive mask
- PCGrad
- CGC / complex expert routing
- LDS
- `loss_dist`

这些模块后续只作为 legacy 能力、消融项或扩展项。

## 重要文件

- `docs/MTL_LITE_DESIGN.md`：新架构、模块边界、接口定义和实施路线。
- `docs/CODEX_CONTEXT.md`：Codex 长期上下文。
- `docs/TODO.md`：分阶段任务计划。
- `docs/BUG_LOG.md`：问题与风险记录。
- `docs/EXPERIMENT_LOG.md`：已完成验证记录。
- `scripts/train.py`：当前训练入口。
- `src/config.py`：配置加载与合并。
- `src/datasets/dataset.py`：数据集与 datamodule。
- `src/models/backbone_factory.py`：通用 backbone factory。
- `src/models/task_heads.py`：通用任务头。
- `src/metrics/metrics.py`：通用指标。

## 已完成验证

- debug smoke 已在服务器环境完整通过。
- `scripts.diagnose` import 和 `--help` 检查已通过。
- 已新增 forward smoke 测试。
- 已新增 regression head backward 非零梯度测试。
- 已新增 loss/metric 一致性测试。
- 已新增 legacy full model README，说明旧模型快照边界。
- 已新增 MTL-Lite 输出 dataclass。
- 已新增 MTL-Lite mask-aware pooling 工具。
- 已新增 `MTLLiteDepressionModel` 骨架。
- 已新增 MTL-Lite forward、backward、config 测试。
- 本地 Codex Python 缺少 `torch`、`pytorch_lightning` 和 `pytest`，MTL-Lite import/pytest 需在服务器训练环境验证。

## 当前优先级

1. 确保 legacy 中的 `local_paths.yaml`、日志、权重、checkpoint 不进入提交。
2. 在服务器训练环境运行 MTL-Lite import 和 pytest 验证。
3. 新增 regression-only 与 MTL-Lite baseline 配置。
4. 设计新主线训练入口或调整现有 runner，使其面向 MTL-Lite。
5. 规划 `src/diagnostics/`，逐步迁移可视化与诊断能力。

## 当前风险

- legacy 不再作为主要维护对象，除非明确要求复现旧模型结果，否则不修复其内部 import。
- 当前工作区可能包含 legacy 迁移中的文件移动或复制，需要避免误删。
- 根目录和 legacy 中的 `local_paths.yaml` 都不应进入 git。
- 旧模型 debug smoke 通过不代表 MTL-Lite 已可运行。
- 诊断逻辑必须避免污染 validation/test。
- 多 GPU DDP metric logging 和 best checkpoint 行为仍需专门验证。
- bf16/mixed precision 稳定性仍需专门验证。

## 推荐验证命令

基础语法检查：

```bash
python -m compileall src scripts tests
```

MTL-Lite import 检查：

```bash
python -c "from src.models.outputs import MTLLiteOutput, MTLLiteLosses; print('outputs import ok')"
python -c "from src.models.temporal.pooling import masked_mean_pool; print('pooling import ok')"
python -c "from src.models.mtl_lite import MTLLiteDepressionModel; print('mtl lite import ok')"
```

legacy 归档检查：

```bash
git diff -- src/legacy/full_model/README.md
```

通用模块检查：

```bash
python -c "from src.models.backbone_factory import build_feature_backbone; print('backbone import ok')"
python -c "from src.models.task_heads import build_regression_task_head; print('task heads import ok')"
python -c "from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss; print('metrics import ok')"
```

MTL-Lite 测试：

```bash
python -m pytest tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py tests/test_mtl_lite_config.py
```

旧主线回归测试仅在需要复现 legacy 行为时运行：

```bash
python -m pytest tests/test_model_forward.py tests/test_loss_backward.py tests/test_loss_metric_consistency.py
```

debug smoke：

```bash
python scripts/train.py --override configs/debug_smoke.yaml
```
