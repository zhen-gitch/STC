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

- [ ] 梳理当前 `src/utils/visualize.py`
- [ ] 新增 `src/diagnostics/`
- [ ] 新增 `src/diagnostics/regression.py`
- [ ] 新增 `src/diagnostics/embeddings.py`
- [ ] 新增 `src/diagnostics/temporal.py`
- [ ] 新增 `src/diagnostics/attribution.py`
- [ ] 新增 `src/diagnostics/reports.py`
- [ ] 支持 `predictions.csv` 导出
- [ ] 支持 prediction-target scatter
- [ ] 支持 residual histogram
- [ ] 支持 BDI 区间误差分析
- [ ] 支持 severity group 误差分析
- [ ] 支持 high-error / low-error subject ranking
- [ ] 支持 t-SNE / UMAP
- [ ] 支持 Grad-CAM / attention CAM / occlusion sensitivity
- [ ] 保留旧诊断入口的向后兼容

## 阶段 7：实验路线

- [ ] 运行 regression-only baseline
- [ ] 运行 MTL-Lite baseline
- [ ] 在相同 split、seed、backbone、时序编码器和指标下比较二者
- [ ] 添加 MTL-Lite + CCC loss 消融
- [ ] 添加 MTL-Lite + LDS 消融
- [ ] 添加 MTL-Lite + `loss_dist` 消融
- [ ] 固定权重 baseline 稳定后，再考虑动态任务权重或梯度冲突处理

## 近期验证

- [ ] 添加 regression head 初始化测试，确保不使用训练标签统计量初始化预测参数
- [ ] 验证 `validation_step` 可以在 `no_grad` 下 forward，且不改变 metrics
- [ ] 确认 LDS label weighting 只使用训练集标签，不接触 val/test 标签
- [ ] 验证 `MODEL_WEIGHT_PATH` 能加载 raw `state_dict` 和 `{"state_dict": ...}` checkpoint
- [ ] 验证多 GPU 下 DDP metric logging 和 best-weight 保存行为
- [ ] 添加 bf16-mixed precision 下预测和 loss finite 测试

## 论文工具

- [ ] 标准化实验日志格式
- [ ] 添加将 `metrics.csv` 导出为 LaTeX 表格的脚本
- [ ] 将诊断图表输出组织为论文可用目录结构

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
