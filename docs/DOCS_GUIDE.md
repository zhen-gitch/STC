# DOCS_GUIDE.md

本文档是项目文档导航与职责分工。后续开始工作时优先读取本文件，用它决定还需要读取哪些文档，避免反复读取所有长文档。

## 当前权威路线

截至 2026-06-25，当前研究路线正式更新为 **RPDF-Net：Risk-aware Progressive De-identification Factorization Network**，中文为 **风险感知递进式去身份因子分解网络**。

当前主线不是继续扩展输入滤镜，也不是只做简单 GRL，而是：

```text
已完成证据层：input artifact / temporal / identity / calibration / geometry audits
        ↓
Stage A: RPDF 证据收口
        layer-wise identity probe + error-identity coupling + artifact weak-label audit
        ↓
Stage B: RPDF-lite 单级因子分解
        H0 -> z_dep, z_m, z_id, z_art, z_res
        ↓
Stage C: 两级递进分解与受控 z_m 传递
        H_k = Phi([z_dep^k, alpha_k * z_m^k])
        ↓
Stage D: 支线有效性验证
        z_art / z_m gate / multi-attacker / severity-balanced loss / dynamic features
```

旧的 input artifact、boundary smoothing、temporal sampling、identity retrieval 和 Shortcut-Regularized MTL 方案现在主要作为 RPDF-Net 的证据基础、对照基线和支线验证，不再作为最终主线。需要决定“接下来做什么”时，优先看 `TODO.md` 的“当前立即执行任务（权威入口）”。

## 快速读取策略

### 只需要了解当前下一步

读取：

1. `docs/DOCS_GUIDE.md`
2. `docs/TODO.md` 开头的 `当前立即执行任务（权威入口）` 段落
3. `docs/CURRENT_STATUS.md` 开头的 `当前权威快照` 段落

### 需要分析 RGB 过拟合机制

读取：

1. `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的 `读者先看：当前研究路线与下一步`、核心判断和三表联合证据
2. `docs/OVERFITTING_MECHANISM_ROADMAP.md` 的机制层级与最新机制更新
3. 必要时读取 `docs/SHORTCUT_AUDIT_DESIGN.md` 的具体脚本/输出规格

### 需要写代码或跑实验

读取：

1. `docs/TODO.md` 的对应任务
2. `docs/EXPERIMENT_SCRIPT_MANUAL.md` 的命令模板
3. 对应设计文档：`MTL_LITE_DESIGN.md`、`SHORTCUT_AUDIT_DESIGN.md` 或 `RGB_OVERFITTING_AUDIT_PLAN.md`

### 需要写论文/解释结果

读取：

1. `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
2. `docs/OVERFITTING_MECHANISM_ROADMAP.md`
3. `docs/RESEARCH_NOTES.md`
4. `docs/EXPERIMENT_LOG.md` 中相关实验条目

## 文档职责划分

| 文档 | 主要职责 | 不应承担的内容 |
|---|---|---|
| `DOCS_GUIDE.md` | 文档导航、读取顺序、职责边界、去重规则 | 详细实验结果、长篇论文分析 |
| `CURRENT_STATUS.md` | 当前状态快照、最新阶段性结论、最近一次决策 | 详细脚本规格、完整历史日志、论文文献展开 |
| `TODO.md` | 可执行任务清单、完成状态、下一步行动项 | 长篇实验解释、重复粘贴完整表格 |
| `EXPERIMENT_LOG.md` | 按时间追加的实验/实现记录，保留历史证据 | 当前权威路线、任务优先级争论 |
| `RGB_OVERFITTING_AUDIT_PLAN.md` | RGB 过拟合研究主控文档，解释实验优先级和论文叙事 | 低层代码接口细节、所有历史命令 |
| `OVERFITTING_MECHANISM_ROADMAP.md` | 上层机制地图、层级模型、决策树、停止规则 | 每次实验的完整数值表、脚本参数 |
| `SHORTCUT_AUDIT_DESIGN.md` | 诊断脚本、输出字段、audit 设计规格 | 当前状态总结、论文长篇叙事 |
| `EXPERIMENT_SCRIPT_MANUAL.md` | 常用运行命令和脚本调用模板 | 实验结果解释、机制判断 |
| `MTL_LITE_DESIGN.md` | MTL-Lite 架构、模块边界、训练入口设计 | RGB 过拟合最新审计结论 |
| `RESEARCH_NOTES.md` | 相关论文、研究背景、可引用理论依据 | 当前任务状态、脚本运行细节 |
| `BUG_LOG.md` | bug、风险、修复记录 | 实验路线规划 |
| `CODEX_CONTEXT.md` | Codex 长期上下文与工作约束，尽量保持紧凑 | 重复粘贴所有实验结果和长表格 |
| `CODEX_PROMPT_TEMPLATES.md` | 可复用提示词模板 | 项目状态和实验结论 |

## 去重写作规则

1. **完整结论只写一次**：
   - 当前阶段性结论写入 `CURRENT_STATUS.md`；
   - 机制解释写入 `RGB_OVERFITTING_AUDIT_PLAN.md` 或 `OVERFITTING_MECHANISM_ROADMAP.md`；
   - 其他文档只用一句摘要并链接到权威文档。

2. **数值表不重复粘贴**：
   - 完整表格优先保留在 `logs/*_summary/tables/*.csv`；
   - 文档只记录最关键的 3-5 个数值和结论。

3. **任务和计划分离**：
   - 可执行任务写 `TODO.md`；
   - 为什么这么做写 `RGB_OVERFITTING_AUDIT_PLAN.md`；
   - 实验执行后发生了什么写 `EXPERIMENT_LOG.md`。

4. **脚本规格和命令分离**：
   - 脚本输入/输出字段写 `SHORTCUT_AUDIT_DESIGN.md`；
   - 快速运行命令写 `EXPERIMENT_SCRIPT_MANUAL.md`。

5. **长期上下文保持精简**：
   - `CODEX_CONTEXT.md` 只保留当前路线、关键约束和最新结论摘要；
   - 避免把 `CURRENT_STATUS.md` 或 `EXPERIMENT_LOG.md` 的完整段落复制进去。

## 当前推荐阅读入口

当前项目重点是 RGB 过拟合机制研究。推荐阅读顺序：

```text
DOCS_GUIDE.md
-> TODO.md 当前立即执行任务
-> CURRENT_STATUS.md 最新段落
-> RGB_OVERFITTING_AUDIT_PLAN.md RPDF-Net 主线
-> OVERFITTING_MECHANISM_ROADMAP.md 机制地图
```

若只是继续执行代码任务，通常不需要完整读取 `EXPERIMENT_LOG.md`、`RESEARCH_NOTES.md` 和 `SHORTCUT_AUDIT_DESIGN.md`。