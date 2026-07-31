# DOCS_GUIDE.md

本文档是项目文档导航与职责分工。后续开始工作时优先读取本文件，用它决定还需要读取哪些文档，避免反复读取所有长文档。

## 当前权威路线

截至2026-07-31，当前主动模型路线正式收敛为 **AU-only Global-Local Alignment（GLA）**：完整global face与眼眉、鼻颊、嘴部三个语义local共享一个backbone，在帧级mask-aware融合后进入时序编码器，以BDI回归为唯一主任务，并只在train阶段使用P0E合格的分组/跨区域AU辅助目标。

此前的identity-adversarial、severity、Stage C task-nuisance和各类artifact实验均作为问题证据、历史基线与外部审计保留，不装入`GLA-FULL`。当前`GLA-FULL`不含ordinal分类、identity-adversarial、Stage C、head-motion或gaze。辅助AU损失下降也不能单独证明抑郁泛化改善或身份信息已被删除，结论仍须由外部identity、severity、task、quality和多seed审计支持。

```text
历史证据层：input artifact / temporal / identity / calibration / geometry / Stage B-C
        ↓（只作为外部审计与匹配基线，不进入GLA-FULL）
当前主动干预：face-valid片段 + 三语义区global/local共享backbone + AU-only
        eye-brow / nose-cheek / mouth-lower-face + global
        -> one shared backbone -> shared projection -> global-residual validity-aware fusion
        -> one-layer GRU -> masked temporal mean
        -> BDI regression + 最大P0E-eligible AU profile（train only）
        -> GLA-FULL优先 -> 依赖感知减法 -> 幸存因素有限加法复核

当前数据门禁：PB-P0A3权威full-rich v2已完成300视频/493,141行
        -> PB-P0B-COMPAT已由commit 485b176实现并推送（70项回归通过）
        -> A2 full v3重新判定PASS（0 blocker / 0 warning；decision a669a35b...e120）
        -> AU-only P0B core至P0E/扩展AU/crop manifest -> 尚未运行或授权真实数据smoke/训练
        head-motion/gaze explicitly excluded; validation/test不读取AU target

输入职责：预处理只做几何、裁切、validity、质量评估与provenance，不做曝光/颜色像素处理
        -> train Dataset可做方向感知曝光增强与普通ColorJitter
        -> 各视图独立采样、单个视图整段序列时间一致；validation/test关闭
        -> 空间flip/affine仍跨视图与时间同步
```

旧的RPDF-Net、`z_art`、`z_m`、递进分解、多级门控、identity-adversarial和Stage C方案保留为历史设计与实验背景，不再作为当前主线或远期默认扩展。input artifact、boundary smoothing、temporal sampling、identity retrieval和Shortcut-Regularized MTL主要作为问题证据、对照基线和审计工具。需要决定“接下来做什么”时，优先看`TODO.md`的“当前立即执行任务（权威入口）”。

### 一屏摘要

| 问题 | 当前决定 |
|---|---|
| 项目目标 | 可审计、可证伪的AU-only global/local行为表征 |
| 结构假设 | `global + 3 local -> shared backbone -> frame fusion -> temporal -> BDI`，区域/跨区域AU只作train-only辅助监督 |
| 明确不含 | ordinal、identity-adversarial、Stage C latent、head-motion、gaze |
| 历史模型角色 | MTL-Lite、Stage B/C、severity和artifact路线只作证据、匹配基线与外部审计 |
| 训练监督 | 规划采用分级区域AU训练期辅助监督；现有PB-P0代码只覆盖历史AU12/14/15+head来源，full-rich来源和P0B兼容代码已完成，但AU-only扩展合同、P0B core、P0C-P0E、模型和训练均未实施或运行 |
| artifact / context / quality | 用于 audit、probe、case study 和 group-wise evaluation |
| AU使用边界 | 核心AU12/14/15，扩展A为AU4/6/7，扩展B为AU10/17；AU6跨眼眉+鼻颊，AU14跨鼻颊+嘴部；gaze和首轮AU25/26排除 |
| 光度边界 | 预处理不改曝光/颜色像素；仅train按view独立、按序列一致地做方向感知曝光增强与普通ColorJitter；val/test关闭 |
| 实验顺序 | 最大合格`GLA-FULL` seed42优先筛选 -> `-B/-A/-all AU/-locals` -> 幸存组单加回与至多一个reduced rebuild -> paired multi-seed |
| 主张边界 | 辅助损失收敛不等于语义解耦；结论必须由外部审计支持 |
| 反证条件 | 不优于paired-seed `GLA-C-REF`、multi-seed不稳定或风险下降以utility恶化为代价时停止扩展 |
| 当前下一步 | 先独立披露并授权P0B core运行；A2 v3 PASS不自动授权P0B，模型真实数据接入仍被AU-only P0E和crop manifest阻断 |
| 后续路线 | P0B core运行 -> AU P0C/P0D/P0E + 三语义区crop/扩展AU合同 -> 独立默认关闭实现 -> FULL优先混合消融 |
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

### 需要审阅AU训练期特权监督方案

读取：

1. `docs/GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`（当前权威训练设计）
2. `docs/PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md`（现有AU12/14/15+head历史来源/P0基础设施；当前只继承AU证据）
3. `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 的主动干预摘要
4. 若要继续P0B-P0E、扩展AU合同或未来模型实现，再读取`docs/MTL_LITE_DESIGN.md`和`docs/TODO.md`

### 需要分析视频有效片段、无效片段与长视频切片

读取：

1. `docs/AVEC2014_SOURCE_DATA_QUALITY.md`
2. `docs/VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md`
3. `docs/CURRENT_STATUS.md` 的最新face-valid slicing摘要
4. 需要运行脚本时再读 `docs/EXPERIMENT_SCRIPT_MANUAL.md`

### 需要写代码或跑实验

读取：

1. 根`AGENTS.md`的`Mandatory pre-implementation disclosure and authorization`
2. `docs/TODO.md` 的对应任务
3. GLA任务先读`docs/GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`；只有Stage C历史复现才读`docs/STAGE_C_RUNBOOK.md`
4. `docs/EXPERIMENT_SCRIPT_MANUAL.md` 的命令模板
5. 其他对应设计文档：`MTL_LITE_DESIGN.md`、`SHORTCUT_AUDIT_DESIGN.md` 或 `RGB_OVERFITTING_AUDIT_PLAN.md`

代码、配置、测试、数据生成、smoke、正式实验、commit和push是互不传递的授权边界。agent必须先提交具名package ID/version及边界、架构/数据流、tensor/mask/loss、文件/API、split/test访问、命令/算力/输出、验证/风险和agent分工，结束该轮并等待用户明确授权；初始“开始/执行”不能跳过首次披露。

### 需要写论文/解释结果

读取：

1. `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
2. `docs/OVERFITTING_MECHANISM_ROADMAP.md`
3. `docs/RESEARCH_NOTES.md`
4. `docs/EXPERIMENT_LOG.md` 中相关实验条目

注意：`RESEARCH_NOTES.md`和`EXPERIMENT_LOG.md`含有RPDF-Net、task-nuisance与旧AU/head历史素材。引用当前路线时必须以本文、`GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md`和`TODO.md`顶部的AU-only GLA说明为准。

## 文档职责划分

| 文档 | 路线状态 | 主要职责 | 不应承担的内容 |
|---|---|---|---|
| `DOCS_GUIDE.md` | 当前导航入口 | 文档导航、读取顺序、职责边界、去重规则 | 详细实验结果、长篇论文分析 |
| `CURRENT_STATUS.md` | 当前状态权威 | 当前状态快照、最新阶段性结论、最近一次决策 | 详细脚本规格、完整历史日志、论文文献展开 |
| `TODO.md` | 当前执行权威 | 可执行任务清单、完成状态、下一步行动项 | 长篇实验解释、重复粘贴完整表格 |
| `RGB_OVERFITTING_AUDIT_PLAN.md` | 当前机制主控 | RGB 过拟合研究主控文档，解释实验优先级和论文叙事 | 低层代码接口细节、所有历史命令 |
| `OVERFITTING_MECHANISM_ROADMAP.md` | 当前机制地图 | 上层机制地图、层级模型、决策树、停止规则 | 每次实验的完整数值表、脚本参数 |
| `AVEC2014_SOURCE_DATA_QUALITY.md` | 当前数据问题权威 | 原视频/aligned/landmark问题、曝光与片段选择依据、数据版本边界 | 模型结构、训练结果历史 |
| `VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md` | 当前输入切片设计 | face-valid片段定义、全片段利用、video-level聚合、AU语义保持的landmark局部增强 | 原始问题长表、完整运行历史 |
| `GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md` | 当前global/local+AU训练设计权威；仅文档、未实施 | 三语义区、共享backbone、跨区域AU、FULL优先混合消融、指标、停止条件和实施路线 | 既有P0代码已经支持的声明、历史运行日志 |
| `PRIVILEGED_BEHAVIOR_ALIGNMENT_PLAN.md` | 现有PB-P0 AU12/14/15+head来源/代码历史权威；当前GLA只继承AU证据 | 历史AU/head来源合同、P0数据门禁、pilot和既有实现边界；说明AU-only P0E覆盖关系 | 把历史head/pose写成当前GLA组件、三语义区未来训练结构、扩展AU已实现的声明 |
| `SHORTCUT_AUDIT_DESIGN.md` | 当前诊断规格 | 诊断脚本、输出字段、audit 设计规格 | 当前状态总结、论文长篇叙事 |
| `STAGE_C_RUNBOOK.md` | Stage C历史复现权威 | 历史信息分流接口、配置矩阵、seed、审计协议、停止条件和实施命令 | 当前GLA任务、Stage B历史结果、长篇文献论证 |
| `EXPERIMENT_SCRIPT_MANUAL.md` | 当前命令手册 | 常用运行命令和脚本调用模板 | 实验结果解释、机制判断 |
| `CODEX_CONTEXT.md` | 当前长期上下文 | Codex 长期上下文与工作约束，尽量保持紧凑 | 重复粘贴所有实验结果和长表格 |
| `MTL_LITE_DESIGN.md` | 历史架构背景与通用组件边界 | MTL-Lite架构、模块边界、训练入口设计；说明独立GLA不得继承的边界 | 当前GLA权威结构、RGB过拟合最新审计结论 |
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
-> AVEC2014_SOURCE_DATA_QUALITY.md 原始/派生数据问题
-> VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md face-valid片段与AU语义保持的landmark局部增强
-> GLOBAL_LOCAL_AU_EXPERIMENT_PLAN.md 三语义区共享backbone与跨区域AU方案
-> RGB_OVERFITTING_AUDIT_PLAN.md task-nuisance 主线
-> OVERFITTING_MECHANISM_ROADMAP.md 机制地图
```

若只是继续执行代码任务，通常不需要完整读取 `EXPERIMENT_LOG.md`、`RESEARCH_NOTES.md` 和 `SHORTCUT_AUDIT_DESIGN.md`。
