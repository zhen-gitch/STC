# DOCS_GUIDE.md

本文档是项目文档导航与职责分工。后续开始工作时优先读取本文件，用它决定还需要读取哪些文档，避免反复读取所有长文档。

## 当前权威路线

截至 2026-07-10，项目目标正式设定为 **Auditable and Falsifiable Coarse-Grained Task-Nuisance Information Separation**，中文为 **可审计、可证伪的粗粒度任务-干扰信息分流**。该目标继承此前 Shortcut-aware Task-Nuisance 路线，但把“语义解耦”从预设结论降为待检验假设。

当前主线不再尝试显式枚举并分解所有潜在因子，例如 `ctx`、`art`、`pose`、`quality` 等。新的原则是：以 `H0 -> z_dep, z_nuisance` 作为粗粒度信息分流假设，只对可验证捷径变量进行弱监督或对抗约束，并通过外部 probe、multi-attacker、leakage matrix、group-wise evaluation 和多 seed 对照审计其效果。reconstruction、decorrelation、训练内 adversary 变弱或单一指标改善，都不能单独证明语义解耦成功。

```text
已完成证据层：input artifact / temporal / identity / calibration / geometry audits
        ↓
Stage A: Shortcut evidence closure
        layer-wise identity probe + error-identity coupling + shortcut/artifact audits as evaluation
        ↓
Stage B: Identity-adversarial task representation
        H0 -> z_dep, with verifiable shortcut suppression
        ↓
Stage C: Coarse task-nuisance information separation
        H0 -> z_dep, z_nuisance
        post-C3 only if re-authorized: H0 -> z_dep, z_id, z_nuisance
        ↓
Stage D: Falsification and robustness validation
        multi-attacker / leakage matrix / severity-balanced loss / group-wise robustness / task consistency
        ↓
当前主动干预：单模型 AU 语义局部输入正则
        dynamic tracking read-only audit -> shared-model training -> global-only inference
```

旧的 RPDF-Net、`z_art`、`z_m`、递进分解和多级门控方案保留为历史设计背景或可选远期支线，不再作为当前主线。input artifact、boundary smoothing、temporal sampling、identity retrieval 和 Shortcut-Regularized MTL 方案现在主要作为问题证据、对照基线和审计工具。需要决定“接下来做什么”时，优先看 `TODO.md` 的“当前立即执行任务（权威入口）”。

### 一屏摘要

| 问题 | 当前决定 |
|---|---|
| 项目目标 | 可审计、可证伪的粗粒度 task-nuisance 信息分流 |
| 结构假设 | `H0 -> z_dep, z_nuisance`，而不是细粒度全因子分解 |
| 显式 latent | 第一版仅 `z_dep`、`z_nuisance`；`z_id` 需基础方案通过 C3 后重新论证 |
| 不显式划分 | `z_art`、`z_ctx`、`z_pose`、`z_quality` 等难以穷尽的因素 |
| 训练监督 | 只对 subject identity 等可验证 shortcut 使用弱监督或对抗约束 |
| artifact / context / quality | 用于 audit、probe、case study 和 group-wise evaluation |
| 主张边界 | 辅助损失收敛不等于语义解耦；结论必须由外部审计支持 |
| 反证条件 | 不优于 paired-seed `C-REF`、multi-seed 不稳定或风险下降以 utility 恶化为代价时停止扩展 |
| 当前下一步 | `AU-T0a` 已通过；运行并人工审阅 `AU-T0b` aligned-coordinate contract -> `AU-T1/T2` 四整体区域动态跟踪审计；通过后才实现单模型训练 |
| 下一步入口 | `TODO.md` 的“当前立即执行任务（权威入口）” |

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
2. Stage C 任务先读 `docs/STAGE_C_RUNBOOK.md`
3. `docs/EXPERIMENT_SCRIPT_MANUAL.md` 的命令模板
4. 其他对应设计文档：`MTL_LITE_DESIGN.md`、`SHORTCUT_AUDIT_DESIGN.md` 或 `RGB_OVERFITTING_AUDIT_PLAN.md`

### 需要写论文/解释结果

读取：

1. `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
2. `docs/OVERFITTING_MECHANISM_ROADMAP.md`
3. `docs/RESEARCH_NOTES.md`
4. `docs/EXPERIMENT_LOG.md` 中相关实验条目

注意：`RESEARCH_NOTES.md` 和 `EXPERIMENT_LOG.md` 含有 RPDF-Net 历史素材。引用时必须以本文和 `CURRENT_STATUS.md` / `TODO.md` 的当前 task-nuisance 路线为准。

## 文档职责划分

| 文档 | 路线状态 | 主要职责 | 不应承担的内容 |
|---|---|---|---|
| `DOCS_GUIDE.md` | 当前导航入口 | 文档导航、读取顺序、职责边界、去重规则 | 详细实验结果、长篇论文分析 |
| `CURRENT_STATUS.md` | 当前状态权威 | 当前状态快照、最新阶段性结论、最近一次决策 | 详细脚本规格、完整历史日志、论文文献展开 |
| `TODO.md` | 当前执行权威 | 可执行任务清单、完成状态、下一步行动项 | 长篇实验解释、重复粘贴完整表格 |
| `RGB_OVERFITTING_AUDIT_PLAN.md` | 当前机制主控 | RGB 过拟合研究主控文档，解释实验优先级和论文叙事 | 低层代码接口细节、所有历史命令 |
| `OVERFITTING_MECHANISM_ROADMAP.md` | 当前机制地图 | 上层机制地图、层级模型、决策树、停止规则 | 每次实验的完整数值表、脚本参数 |
| `SHORTCUT_AUDIT_DESIGN.md` | 当前诊断规格 | 诊断脚本、输出字段、audit 设计规格 | 当前状态总结、论文长篇叙事 |
| `STAGE_C_RUNBOOK.md` | 当前 Stage C 实施权威 | 信息分流接口、配置矩阵、seed、审计协议、停止条件和实施命令 | Stage B 历史结果、长篇文献论证 |
| `EXPERIMENT_SCRIPT_MANUAL.md` | 当前命令手册 | 常用运行命令和脚本调用模板 | 实验结果解释、机制判断 |
| `CODEX_CONTEXT.md` | 当前长期上下文 | Codex 长期上下文与工作约束，尽量保持紧凑 | 重复粘贴所有实验结果和长表格 |
| `MTL_LITE_DESIGN.md` | 架构背景 | MTL-Lite 架构、模块边界、训练入口设计 | RGB 过拟合最新审计结论 |
| `RESEARCH_NOTES.md` | 文献与历史理论素材 | 相关论文、研究背景、可引用理论依据 | 当前任务状态、脚本运行细节 |
| `EXPERIMENT_LOG.md` | 历史日志 | 按时间追加的实验/实现记录，保留历史证据 | 当前权威路线、任务优先级争论 |
| `BUG_LOG.md` | 风险记录 | bug、风险、修复记录 | 实验路线规划 |
| `CODEX_PROMPT_TEMPLATES.md` | 辅助材料 | 可复用提示词模板 | 项目状态和实验结论 |

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
-> RGB_OVERFITTING_AUDIT_PLAN.md task-nuisance 主线
-> OVERFITTING_MECHANISM_ROADMAP.md 机制地图
```

若只是继续执行代码任务，通常不需要完整读取 `EXPERIMENT_LOG.md`、`RESEARCH_NOTES.md` 和 `SHORTCUT_AUDIT_DESIGN.md`。
