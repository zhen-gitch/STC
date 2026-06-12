# Legacy Full Model

本目录保存旧版大型端到端模型的历史快照。

## 定位

`src/legacy/full_model/` 只用于保留旧模型能力和历史实验上下文，不再作为当前论文主线继续开发。

当前论文主线已经切换为 MTL-Lite 轻量级多任务 BDI 预测模型。新模型应在主线 `src/models/` 下独立实现，不应继承或扩展 legacy 中的旧 `EndToEndDepressionModel`。

## 包含内容

legacy full model 可包含：

- 旧版 `EndToEndDepressionModel`
- CGC / expert routing
- contrastive learning
- adaptive mask
- PCGrad
- uncertainty weighting
- LDS label weighting
- `loss_dist`
- 旧版 runner、脚本、配置和测试快照
- 旧版可视化 hooks

这些能力后续只作为历史参考、回退方案或消融灵感，不作为新主线默认依赖。

## 维护策略

- 不主动修复 legacy 内部 import。
- 不主动重构 legacy 内部代码。
- 不把 MTL-Lite 新逻辑加入 legacy。
- 不把 legacy runner 接回新主线训练入口。
- 只有在用户明确要求复现旧模型结果时，才针对 legacy 做最小必要修复。

## 隐私与提交限制

不要提交以下内容：

- `local_paths.yaml`
- 数据集
- 日志
- checkpoint
- 权重文件
- 私有凭证
- 实验输出

如需在 legacy 快照中运行旧模型，请在本机或服务器上自行准备私有路径配置，并确保该配置不进入 git。

## 新主线位置

新模型和通用组件应放在：

```text
src/models/
src/metrics/
src/datasets/
src/diagnostics/
```

MTL-Lite 相关实施路线见：

```text
docs/MTL_LITE_DESIGN.md
docs/CODEX_CONTEXT.md
docs/TODO.md
```
