# RESEARCH_NOTES.md

本文档记录与当前论文主线直接相关的研究背景、论文线索和后续实验方向。项目后续默认以中文维护研究笔记，论文题名、模型名、数据集名和链接保留英文。

## 2026-07-06 路线修订说明

当前主线已从细粒度 **RPDF-Net** 调整为 **Shortcut-aware Task-Nuisance Disentangled Representation Learning**。因此，本文中关于 `z_dep/z_m/z_id/z_art/z_res`、RPDF-lite、两级递进和受控 `z_m` 的段落保留为历史理论素材，不再作为当前模型实现路线。当前应优先引用以下表述：

```text
H0 -> z_dep, z_nuisance
optional: H0 -> z_dep, z_id, z_nuisance
```

artifact、context、pose、quality 等因素只作为审计变量、post-hoc probe、case study 和 group-wise evaluation，不作为第一版显式 latent。

## 当前问题判断

当前项目使用的是经过 OpenFace 裁剪和对齐后的人脸视频帧序列。因此，模型失效风险不应简单理解为“原始背景过拟合”，而应更准确地表述为：

> 模型可能在 OpenFace aligned face 中学习了身份、纹理、裁剪边界、对齐伪影、姿态残留、追踪质量、光照和视频质量等非抑郁捷径，而不是跨 subject 稳定的面部行为动态。

已经观察到的现象包括：

- regression-only baseline 在训练集上可以持续拟合，但验证集和测试集泛化不稳定；
- 冻结 backbone 底层、只微调最后 1 或 2 个 transformer blocks 后，并未明显改善 test 表现；
- last1/last2 结果提示问题不只是 backbone 可训练层数，而更可能是输入表征和监督信号没有充分约束模型关注抑郁相关面部行为；
- 当前 ordinal BDI 辅助任务本质上仍来自同一个 BDI 标签，可能不足以强迫模型学习 AU、landmark motion、gaze、pose 等行为线索。

## 当前主线：Task-Nuisance 理论支撑与研究定位

截至 2026-07-06，当前研究主线应表述为 **Shortcut-aware Task-Nuisance Disentangled Representation Learning（捷径感知的任务-干扰粗粒度解耦表征学习）**。它不是简单把输入滤镜、GRL、重加权和动态特征堆在一起，也不再尝试把所有潜在因素逐项显式分解；而是把已有实验中反复出现的过拟合现象整理成一个统一问题：静态 RGB aligned face 同时包含抑郁相关面部线索、身份线索、OpenFace artifact/domain proxy、severity 分布偏置和任务语境混杂。当前模型只显式学习 `z_dep` 和 `z_nuisance`，必要时加入可验证的 `z_id` 出口。

理论依据可以按以下几条线组织：

| 理论线索 | 对本项目的支撑 | 当前路线中的落点 |
|---|---|---|
| Shortcut learning | 深度模型会优先学习身份、采集条件、裁剪边界、质量差异等容易但不可迁移的线索 | 先做 layer-wise identity probe、error-identity coupling 和 artifact weak-label audit，再进入模型干预 |
| Disentanglement / factorization | 端到端共享 embedding 难以解释，但过细因子不可验证 | 只做 `z_dep/z_nuisance` 粗粒度分流，可选 `z_id` |
| Domain-adversarial / privacy-preserving representation | 对抗分支可降低表征中 subject identity 的可用性，但可能误伤有效行为线索 | subject-adversarial GRL 作为 Stage B 必要对照 |
| Privacy-utility trade-off | 完全去身份可能损失年龄、表情基线、面部活动幅度等与抑郁相关的交叠信息 | 不显式建 `z_m`，通过 BDI utility、identity risk 和 nuisance leakage 联合评估 |
| Imbalanced regression / long-tail severity | BDI 分数段不均会使多数分段主导梯度，造成 minimal/severe 系统偏置 | severity-balanced regression 作为对照或支线 |
| Facial behavior depression literature | 抑郁线索更多体现在表情活动、AU、landmark motion、pose/gaze 等行为模式中 | dynamic / behavior features 暂列 Stage D |
| OpenFace artifact and occlusion literature | aligned crop、黑边、麦克风、眼镜、胡须、追踪失败和几何尺度都会产生稳定非抑郁线索 | artifact/context/quality 仅作为 audit、probe 和 group-wise evaluation |

当前实验推进逻辑：

```text
已完成证据层：input artifact / temporal / identity / calibration / geometry audits
-> Stage A: shortcut 证据收口
-> Stage B: identity-adversarial task representation
-> Stage C: coarse task-nuisance disentanglement
-> Stage D: robustness validation 与行为动态扩展
```

因此，后续论文叙事应强调“从过拟合机制审计到粗粒度稳健表征学习”的连续性：已有消融不是零散失败记录，而是在逐步证明单一输入处理无法解释全部泛化问题，从而引出 task-nuisance 解耦作为更可验证的解决框架。

## 历史理论素材：RPDF-Net 路线的必要性与可行性

本节保留为历史理论素材。下面关于 RPDF-Net、`z_m`、`z_art` 和多级递进的论证不再是当前实现路线，但其中关于 shortcut learning、信息瓶颈、可辨识性、隐私-效用权衡和 severity imbalance 的文献仍可支撑当前 task-nuisance 主线。

### 1. Shortcut learning 支持“先审计、再建模”

Shortcut learning 研究指出，深度模型会优先使用训练分布中稳定、容易学习但不可迁移的线索。对本项目而言，OpenFace aligned face 中的黑边、裁剪硬边界、麦克风遮挡、眼镜反光、胡须、几何尺度、视频长度和任务语境，都可能成为比面部行为更容易利用的 shortcut。

参考：

- Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780

对 RPDF-Net 的含义：Stage A 必须先做 layer-wise identity probe、error-identity coupling 和 artifact weak-label audit。否则直接改模型，只能得到“某个实验指标变好/变坏”，无法证明模型是否真的减少了非抑郁捷径。

### 2. 小样本 ViT/DeiT 风险支持 patch-level shortcut 审计

当前 RGB backbone 属于 patch-based transformer。小样本视觉 Transformer 通常更依赖数据规模、正则化和监督约束；当输入为 `112 x 112` 且 patch size 为 `16` 时，一帧只有约 `7 x 7` 个 patch，黑边、局部遮挡、眼镜、胡须和裁剪残留都可能占据完整 patch。

参考：

- Efficient Training of Visual Transformers with Small Datasets: https://arxiv.org/abs/2106.03746
- Training data-efficient image transformers & distillation through attention: https://arxiv.org/abs/2012.12877

对 RPDF-Net 的含义：问题不应只归因于“上层 head 参数过多”或“backbone 微调层数不对”。更合理的做法是让 backbone 保留面部表征能力，同时在上层显式审计和分流 identity/artifact/severity 风险。

### 3. 信息论路径：信息瓶颈与 minimal sufficient representation（z_dep 去身份主线）

"图像表征保留目标信息、去除无关信息"在信息论上有成熟形式化：`max I(Z;Y) - β·I(Z;X)`——保留与目标 Y 的互信息，压缩与输入 X 的互信息；"无关信息"= 与 Y 无关的 X 信息。

#### 3.1 理论根基：Achille-Soatto 信息论框架

- **核心定理**：对 nuisance 因子的不变性 ≡ 学到表征的信息极小性。即去无关信息不必显式对抗——对 z_dep 施加信息瓶颈压缩，不变性作为副产品涌现，且比对抗训练更稳。
- **训练动力学**：语义有意义但最终无关的信息在训练早期被编码，随后才被丢弃——这解释了 RGB 过拟合机制：训练不足时残留无关身份/伪迹信息。

参考（均已 peer-review 核验）：

- Information Dropout: Learning Optimal Representations Through Noisy Computation（**IEEE TPAMI 2016**，Achille-Soatto，不变性≡信息极小性）：https://arxiv.org/abs/1611.01353
- Emergence of Invariance and Disentanglement in Deep Representations（Achille-Soatto，信息极小→不变性涌现）：https://arxiv.org/abs/1706.01350
- Usable Information and Evolution of Optimal Representations During Training（**ICLR 2021**，早期编码-后期丢弃）：https://arxiv.org/abs/2010.02459

#### 3.2 信息瓶颈实现谱系

- Deep Variational Information Bottleneck（**ICLR 2017**，Alemi et al.）：变分近似 IB + reparameterization trick，奠基深度 IB，泛化与对抗鲁棒性优于传统正则：https://arxiv.org/abs/1612.00410
- Cell Variational Information Bottleneck Network（**ACML 2023**，cellVIB）：逐层 VIB 单元，正则随层渐增（vs Deep VIB 全压输出层）——Stage C 逐层净化的机制依据：https://arxiv.org/abs/2403.15082
- Dynamic Multimodal Information Bottleneck（**WACV 2024**）：sufficiency loss 显式防止 IB 压缩丢弃任务相关信息——防 z_dep 误丢抑郁信号：https://arxiv.org/abs/2311.01066

#### 3.3 minimal sufficient 警告

对比学习能近似得到 minimal sufficient representation，但对下游任务**不充分**——非共享的任务相关信息会被忽略。这直接警告 RPDF：单纯压缩 z_dep 可能让它丢掉身份-抑郁交叠部分的真实抑郁信号，必须配合 sufficiency loss。

参考（已 peer-review 核验）：

- Rethinking Minimal Sufficient Representation in Contrastive Learning（**CVPR 2022**）：https://arxiv.org/abs/2203.07004

对 RPDF-Net 的含义：`z_dep` 去身份默认路径为 **IB 压缩**（`max I(z_dep;BDI) - β·I(z_dep;H0)`）+ **sufficiency loss** 保 BDI 信息。GRL 对抗降为对照支线，不作主线。

### 4. 表征解耦与可辨识性：z_m 必须有显式监督

面部表情识别、隐私保护表征和 disentanglement 研究支持把任务相关信息和身份、姿态、域因素分开建模。但 Locatello 等的不可辨识定理从理论上证明：**无监督的 disentanglement 在没有归纳偏置时根本不可辨识**——训了 12000+ 模型，无监督下无法识别出 well-disentangled 模型。后续所有可辨识工作都靠显式监督或归纳偏置。

参考（均已 peer-review 核验）：

- Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations（**ICML 2019**，Locatello et al.，1792 引用，不可辨识定理）：https://arxiv.org/abs/1811.12359
- A Sober Look at the Unsupervised Learning of Disentangled Representations（**JMLR 2020**，扩展版）：https://arxiv.org/abs/2010.14766
- Leveraging sparse and shared feature activations for disentangled representation learning（**NeurIPS 2023**，Fumero et al.，多任务+稀疏可辨识）：https://arxiv.org/abs/2304.07939
- Learning Disentangled Representations via Mutual Information Estimation（**ECCV 2020**，shared/exclusive 分解，z_m 拓扑同构）：https://arxiv.org/abs/1912.03915

对 RPDF-Net的直接含义：`z_m` 在原 B1 设计中**无显式监督**（仅"受控进入预测"），恰好踩在不可辨识定理的雷上——模型可把身份任意塞进 z_m 或 z_dep，因子分解失效。因此 **`z_m` 必须有显式监督或归纳偏置**。RPDF-Net 不应宣称"实现了 disentanglement"，应定位为"在有监督归纳偏置下的可审计信息分流"。

`z_dep/z_m/z_id/z_art/z_res` 的价值不在于宣称完全可辨识，而在于建立可审计出口：

```text
z_dep: low-risk depression factor, IB-compressed (invariance emerges from info minimality) + sufficiency loss
z_m: identity-depression mutual factor, MUST have explicit supervision (identifiability), controlled use
z_id: subject/static appearance outlet
z_art: OpenFace artifact/domain outlet
z_res: residual outlet with information bottleneck (dim << others), preventing escape
```

第一版应做 RPDF-lite，而不是直接上完整多级多门控结构。

### 4.5 去混淆表征：z_dep⊥z_id 的图像域直接先例

因果推断领域的"去混淆表征"把混淆变量作为单独因子剥离，保留因果相关因子，与 RPDF 的 z_dep⊥z_id 拓扑同构。

参考（均已 peer-review 核验）：

- Backdoor Defense via Deconfounded Representation Learning（**CVPR 2023**）：把后门攻击当混淆，clean 模型通过**最小化与混淆表征的 MI** `min I(z_clean; z_confound)` 捕获因果效应——与 RPDF `min I(z_dep; z_id)` 直接同构，是图像域 peer-reviewed 先例：https://arxiv.org/abs/2303.06818
- Robust Causal Graph Representation Learning against Confounding Effects（**AAAI 2023**，RCGRL）：生成工具变量消除混淆，捕获因果判别信息：https://arxiv.org/abs/2208.08584
- Bounds on Representation-Induced Confounding Bias（**ICLR 2024**）：证明低维表征会丢失观察混淆信息导致偏差，给出非可辨识条件——**警告：z_dep 的 IB 压缩不能无脑降维，压缩过度会丢失 BDI 相关的混淆调整信息，dim 需配合 sufficiency test**：https://arxiv.org/abs/2311.11321

对 RPDF-Net 的含义：z_dep 去身份除 IB 压缩外，可借鉴 CVPR 2023 的 **MI 最小化去混淆** `min I(z_dep; z_id)` 作为第二条主线候选（显式 MI 最小化，非对抗抑制）。同时受 ICLR 2024 警告约束：z_dep 降维须配合 `I(z_dep;BDI)` 下限监控。

### 5. 连续标签不均衡研究支持 severity-balanced 支线

BDI 是连续分数，且 minimal/mild/moderate/severe 分段通常不均衡。Deep Imbalanced Regression 和 Balanced MSE 相关研究说明，连续标签分布不均会使多数分段主导梯度，造成少数分段系统偏置。

参考：

- Delving into Deep Imbalanced Regression: https://arxiv.org/abs/2102.09554
- Balanced MSE for Imbalanced Visual Regression: https://arxiv.org/abs/2203.16427

对 RPDF-Net 的含义：severity-balanced regression 的目标是缓解 score-bin / severity-bin 不均衡，而不是人为提高预测值方差。它应作为 RPDF-Net 的支线验证：若它改善 minimal/severe bias 且不恶化 identity risk、CCC 和 task consistency，则说明 severity imbalance 是独立机制；否则不能把它并入主模型。

### 6. OpenFace/LibreFace 与面部行为研究支持 artifact weak labels 与后续 dynamic 支线

OpenFace/LibreFace/OpenFace 3.0 说明 landmark、AU、pose、gaze、confidence、success 本身就是可量化的面部行为与质量变量。抑郁识别相关研究也强调 AU、landmark temporal dynamics 和面部行为模式的重要性。

参考：

- LibreFace: https://arxiv.org/abs/2308.10713
- OpenFace 3.0: https://arxiv.org/abs/2506.02891
- FacialPulse: https://arxiv.org/abs/2408.03499
- Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units: https://arxiv.org/abs/2407.13753

对 RPDF-Net 的含义：

- `z_art` 可以用 confidence、success、bbox scale、center offset、black-border ratio、edge gradient、landmark jitter 等弱标签进行审计；
- dynamic facial behavior 有理论价值，但应暂列 Stage D。只有当 RPDF-lite 已经降低 identity/artifact risk 而 BDI 表现仍受限时，才启动 dynamic branch。

### 7. 当前路线的论文贡献表达

推荐表述：

```text
We propose a mechanism-audited, risk-aware factorization framework for RGB facial depression prediction. Instead of treating de-identification, artifact mitigation, severity balancing, and behavioral dynamics as independent modules, we first audit how identity, OpenFace artifacts, temporal/task context, and severity imbalance enter the prediction process, and then factorize the shared representation into depression, mutual identity-depression, identity, artifact, and residual factors.
```

中文表达：

```text
本文提出一种机制审计驱动的风险感知因子分解框架。该框架不把去身份、伪迹处理、严重程度重加权和动态特征视作彼此独立的模块堆叠，而是先审计身份、OpenFace 伪迹、时序/任务语境和分数段不均衡如何进入预测过程，再将共享表征分解为抑郁因子、身份-抑郁交叠因子、身份因子、伪迹因子和残差因子。
```

## 因子分解四难题的先例映射与核验结论

因子分解在通用表征学习中存在四个公认难题。下面整理每个难题的 peer-reviewed 先例与 RPDF-Net 借鉴路径。所有引用均经 DBLP/OpenAlex/Semantic Scholar 交叉核验 venue 与引用数。

### 难题 1：竞争分配（信息无唯一归宿）

固定维表征拆成多因子是欠定问题，同一份信息（如眼袋深度既是疲劳/抑郁线索又是个体身份特征）该进哪个因子无唯一解。

**先例**：

- 不可辨识定理（**ICML 2019 + JMLR 2020**，[1811.12359](https://arxiv.org/abs/1811.12359)/[2010.14766](https://arxiv.org/abs/2010.14766)）：无监督下不可辨识，必须靠监督/归纳偏置固定归宿。
- 多任务+稀疏可辨识（**NeurIPS 2023**，[2304.07939](https://arxiv.org/abs/2304.07939)）：多任务访问在 sufficiency+minimality 下足以可辨识。
- shared/exclusive 分解（**ECCV 2020**，[1912.03915](https://arxiv.org/abs/1912.03915)）：显式拆 shared（对应 z_m）+ exclusive（对应 z_dep/z_id），MI 约束定义。
- IB 压缩分配（**ICLR 2017** Deep VIB [1612.00410](https://arxiv.org/abs/1612.00410) + **TPAMI 2016** [1611.01353](https://arxiv.org/abs/1611.01353)）：z_dep 通过 IB 目标 `max I(z_dep;BDI) - β·I(z_dep;H0)` 被压成只含 BDI 必要信息，剩余信息自然流向其他因子——竞争分配由 IB 目标决定，不再欠定。

**RPDF 借鉴**：z_m 必须有显式监督（边际增益监督，借用 ECCV 2020 shared/exclusive），否则它是身份逃生口。用 IB 压缩 + sufficiency loss 决定信息归宿，放弃线性正交（无法处理 z_m 与 z_dep 的非线性相关）。

### 难题 2：排他约束（post-factorization 去身份）

即使 z_dep 被推成 identity-free，z_m 按定义携带身份且参与预测，模型可经 z_m 路由身份进预测，绕过 z_dep 约束。线性正交不防止非线性投影恢复身份。

**先例**：

- MI 最小化去混淆（**CVPR 2023**，[2303.06818](https://arxiv.org/abs/2303.06818)）：clean 模型 `min I(z_clean; z_confound)` 捕获因果效应——与 RPDF `min I(z_dep; z_id)` 拓扑同构，图像域 peer-reviewed 直接先例，非对抗式。
- 抑郁+身份对抗（**INTERSPEECH 2022**，[2206.09530](https://arxiv.org/abs/2206.09530)）：`min depression loss + max speaker loss`，DAIC-WOZ 上做——GRL 对抗的抑郁领域直接对标先例。
- 对抗训练不稳定警告：隐私表征学习中对抗方法存在训练不稳定问题，independence regularization 是更稳替代。

**RPDF 借鉴**：GRL 接 z_dep（post-factorization 拓扑），但默认主线改为 IB 压缩（不变性涌现，无需对抗）+ MI 最小化去混淆（CVPR 2023）。GRL 降为对照基线，对标 INTERSPEECH 2022。

### 难题 3：递进净化（单调性无保证）

Stage C 假设 `H_k = Phi([z_dep^k, alpha_k·z_m^k])` 后 identity risk 随 k 下降，但这是假设不是定理。

**先例**：检索 15 篇 progressive/hierarchical/iterative disentanglement 文献，**无一证明 identity/nuisance 跨阶段单调下降**，全部是 coarse-to-fine 经验策略。

**RPDF 借鉴**：Stage C 降为可证 falsifiable 假设。机制改用逐层 IB（**ACML 2023** cellVIB [2403.15082](https://arxiv.org/abs/2403.15082)，β_k 递增驱动信息单调下降）而非堆叠因子分解层。终止判据写成"`identity_risk(z_dep^k)` 曲线不单调下降则停止"，不默认两级有效。

### 难题 4：外部验证（不可证实性）

因子分解欠定，外部 attacker 是唯一裁判，但只能证伪（"这些攻击者恢复不出身份"），不能证实（"z_dep 含的是正确抑郁信息"）。

**先例**：

- worst-case attribute inference guarantee（**AAAI 2024** TAPPFL [2312.06989](https://arxiv.org/abs/2312.06989)）：对 worst-case 攻击的可证明 guarantee + utility-privacy inherent tradeoff（定理级）。
- 对比学习隐私（**CCS 2021** Talos [2102.04140](https://arxiv.org/abs/2102.04140)）：不同 attacker 给不同风险，必须多攻击者联合报告。
- 攻击降到随机猜测水平（[2007.15064](https://arxiv.org/abs/2007.15064)）：可操作的"成功"阈值。
- minimal sufficient 对下游不充分（**CVPR 2022** [2203.07004](https://arxiv.org/abs/2203.07004)）：补上"只能证伪"缺口的 sufficiency test 正面证据。

**RPDF 借鉴**：外部验证 = 双门——attacker 证伪身份泄漏（必要）+ sufficiency test 证实 BDI 信息保留（充分）。预注册 attacker 集合（kNN+linear+MLP 固定容量），报最坏值；同时打 z_dep/z_m/prediction 三个对象；成功阈值 = attacker accuracy ≈ chance level。接受 utility-privacy 是 frontier 而非单点。

### 核验降权与剔除清单

核验中发现以下文献需降权或剔除，避免被未经验证的非正式发表误导：

- **IRM（Invariant Risk Minimization, [1907.02893](https://arxiv.org/abs/1907.02893)）**：DBLP 仅 CoRR，OpenAlex 无正式 venue DOI——**从未在 peer-reviewed venue 发表**。且 [2101.01134](https://arxiv.org/abs/2101.01134)/[2010.05761](https://arxiv.org/abs/2010.05761) 证明 IRMv1 脆弱、非线性下可灾难性失败。**不作为 RPDF 依据**，OOD 不变性改用 IB 路径。
- **Fair Sufficient Representation Learning（[2504.01030](https://arxiv.org/abs/2504.01030)）**：仅 CoRR 2025 预印本。剔除，其思想由 CVPR 2022 [2203.07004](https://arxiv.org/abs/2203.07004) + AAAI 2024 [2312.06989](https://arxiv.org/abs/2312.06989) 覆盖。
- **IndiSeek（[2509.21584](https://arxiv.org/abs/2509.21584)）**：仅 CoRR 2025 预印本。剔除，independence+completeness 由 NeurIPS 2023 [2304.07939](https://arxiv.org/abs/2304.07939) 覆盖。
- **Rényi Fair IB（[2203.04950](https://arxiv.org/abs/2203.04950)）**：CWIT 2022（小型 workshop）。降为次要参考，不作主要依据。
- **修正声称**：Information Dropout venue 为 **IEEE TPAMI**（非 JMLR）；[2203.07004](https://arxiv.org/abs/2203.07004) 为 CVPR 2022（去掉先前"oral"标记，DBLP 未证实 oral）；RCGRL [2208.08584](https://arxiv.org/abs/2208.08584) 为 AAAI 2023（去掉 oral 标记）。

## OpenFace 版本与数据约定

当前主数据版本来自已有 OpenFace 裁剪对齐流程，所用 OpenFace 版本可能不是最新版。短期内不建议直接升级 OpenFace 并覆盖已有数据，因为这会改变 crop、landmark、AU、pose、confidence 等分布，相当于更换数据版本。

推荐约定：

- 保留当前 OpenFace 版本生成的数据作为主数据版本；
- 明确记录 OpenFace 版本、命令、输出字段、裁剪尺寸和帧采样方式；
- 后续如需使用 OpenFace 3.0、LibreFace 或其他工具，应作为独立数据版本和消融实验；
- 不混用不同 OpenFace 版本生成的帧、landmark 或 AU 特征；
- 若存在 OpenFace 原始 CSV，应优先将 `confidence`、`success`、pose、gaze、AU、landmark 等结构化输出纳入诊断。

## 相关研究线索

### AVEC2014 与面部视频抑郁预测

- AVEC2014 / Audio-Visual Emotion Challenge 是当前项目数据设定的重要参照，BDI-II 连续分数预测是典型任务形式。
- 需要在论文中明确说明 split、subject 独立性、评价指标和是否使用 validation/test 标签参与任何统计。

### Temporal facial landmarks

- FacialPulse, 2024: https://arxiv.org/abs/2408.03499
- 研究动机与当前项目高度相关：端到端图像特征可能包含大量冗余和身份信息，而 temporal facial landmarks 更接近面部行为动态。
- 对本项目启发：应建立 landmark-only temporal baseline，并比较其与 RGB aligned face baseline 的泛化差异。

### 多尺度时序与抑郁相关特征增强

- Two-stage Temporal Modelling Framework, 2021: https://arxiv.org/abs/2111.15266
- 该方向强调短时行为片段、多尺度时序建模和 Depression Feature Enhancement，用于增强抑郁相关线索并抑制非抑郁噪声。
- 对本项目启发：后续可从简单的 clip-level temporal pooling、keyframe weighting、temporal occlusion 开始，不必一开始复现复杂图结构。

### Action Units 与面部行为 biomarker

- Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units, 2024: https://arxiv.org/abs/2407.13753
- 该方向强调 AU、expression、temporal statistics 与抑郁状态之间的关系。
- 对本项目启发：MTL 辅助任务应优先考虑 AU intensity、AU presence、expression distribution、landmark motion、pose/gaze 等行为信号，而不是只使用 BDI ordinal 分箱。

### OpenFace / LibreFace / OpenFace 3.0

- OpenFace 3.0, 2025: https://arxiv.org/abs/2506.02891
- LibreFace, 2023: https://arxiv.org/abs/2308.10713
- OpenFace 相关工具链提供 landmark、AU、head pose、gaze、confidence、success 等结构化面部行为输出。
- 对本项目启发：当前 OpenFace aligned frames 不应只作为图像输入，也应尽量利用 OpenFace CSV 作为诊断特征、辅助监督或轻量行为分支输入。

### 去身份化与捷径学习

- OpticalDR, 2024: https://arxiv.org/abs/2402.18786
- Shortcut Learning in Deep Neural Networks, 2020: https://arxiv.org/abs/2004.07780
- Occlusion-Adaptive Deep Network for Robust Facial Expression Recognition, 2020: https://arxiv.org/abs/2005.06040
- A survey of face recognition techniques under occlusion, 2020: https://arxiv.org/abs/2006.11366
- When Face Recognition Meets Occlusion, 2021: https://arxiv.org/abs/2103.02805
- 这些研究支持一个关键风险判断：深度视觉模型可能优先学习身份、采集条件、质量差异等容易但不可迁移的特征。
- 对本项目启发：需要设计 identity/quality/pose/crop artifact 相关的诊断和消融，而不是只比较 backbone 或训练 epoch。
- 对遮挡因素的启发：眼镜、麦克风、胡须和口鼻周围遮挡会改变局部可见面部区域，既可能破坏表情/AU/landmark 观测，也可能形成稳定 subject 或采集条件标记；因此应作为 occlusion shortcut 审计对象，而不是简单当作噪声删除。

### 多任务损失与负迁移

- GradNorm: https://arxiv.org/abs/1711.02257
- Uncertainty Weighting: https://arxiv.org/abs/1705.07115
- PCGrad: https://arxiv.org/abs/2001.06782
- 这些方法不是当前 task-nuisance 主线的第一优先级，但在引入 AU、landmark、pose、gaze 等辅助任务后，可作为负迁移控制和任务权重消融。

## 历史阶段：早期 behavior / shortcut 实验方向（已被当前证据层继承）

### P0：诊断模型是否学习了非抑郁捷径

详细实施方案见 `docs/SHORTCUT_AUDIT_DESIGN.md`。该框架命名为 Shortcut Audit Framework，目标是在继续修改模型前，先验证 OpenFace aligned face 中的身份、追踪质量、姿态、裁剪伪影、视频质量等非抑郁变量是否与 BDI、预测值、残差或绝对误差存在关系。

1. OpenFace 质量统计
   - 统计每个视频的 `confidence` 均值、方差、低置信帧比例；
   - 统计 `success` 失败帧比例；
   - 统计 pose/gaze 分布和 landmark 抖动；
   - 分析这些变量与 BDI、预测误差、残差、subject 的相关性。

2. 输入消融
   - aligned RGB face；
   - grayscale aligned face；
   - masked face，弱化脸部边界、头发、衣服、裁剪边缘；
   - landmark heatmap；
   - landmark/AU/pose only；
   - 低频或模糊图像，用于判断模型是否依赖细粒度身份纹理。

3. 归因与遮挡分析
   - model attention / Grad-CAM / input-gradient；
   - occlusion sensitivity；
   - keyframe importance；
   - 高误差和低误差 subject case study；
   - 重点检查模型关注区域是否集中在眼、眉、嘴、鼻唇沟，而不是脸部边缘、头发、眼镜、黑边或裁剪伪影。

### P1：建立行为表征 baseline

1. Landmark-only temporal baseline
   - 输入 OpenFace landmark 坐标；
   - 派生速度、加速度、关键区域距离；
   - 使用 GRU、TCN 或轻量 Transformer 做 video-level BDI 回归。

2. AU / pose / gaze baseline
   - 输入 AU intensity / AU presence、head pose、gaze、confidence/success mask；
   - 建立轻量时序模型；
   - 与 RGB baseline 在相同 split、seed、metric 下比较。

3. RGB + behavior late fusion
   - RGB branch 使用现有 MTL-Lite 图像分支；
   - behavior branch 使用 landmark/AU/pose/gaze 序列；
   - video-level representation 后 concat + MLP；
   - 先做 regression-only，再考虑多任务。

### P2：重构多任务学习目标

当前 BDI ordinal 辅助任务可保留为 baseline，但后续更值得尝试的辅助任务包括：

- AU intensity reconstruction；
- AU presence classification；
- landmark motion prediction；
- pose/gaze prediction；
- expression distribution prediction；
- temporal smoothness 或 motion contrast；
- frame quality / OpenFace confidence prediction，仅作为诊断或辅助约束。

引入这些任务后，再考虑 uncertainty weighting、GradNorm、PCGrad 或简单任务权重网格搜索。

## 历史推荐实验顺序

以下顺序是项目早期在 behavior baseline 尚未完成、RGB 输入伪迹尚未充分诊断时的路线。当前已被 `docs/RGB_OVERFITTING_AUDIT_PLAN.md` 中的多因素过拟合审计路线取代，保留在此仅作为历史背景。

1. `E0_openface_quality_correlation`：OpenFace 质量、姿态、gaze、AU 与 BDI/误差相关性。
2. `E1_input_ablation`：RGB aligned、grayscale、masked face、landmark heatmap、landmark/AU/pose only 对照。
3. `E2_landmark_temporal_baseline`：landmark-only 时序回归。
4. `E3_au_pose_gaze_baseline`：AU/pose/gaze-only 时序回归。
5. `E4_rgb_behavior_late_fusion`：图像分支与行为分支后融合。
6. `E5_behavior_auxiliary_mtl`：以 AU、landmark motion、pose/gaze 为辅助任务的多任务模型。
7. `E6_multiscale_temporal`：多尺度 clip/video temporal aggregation。
8. `E7_loss_balancing`：在行为辅助任务稳定后，再做 uncertainty weighting、GradNorm、PCGrad 或任务权重消融。

## 历史结论：多因素 RGB 过拟合审计（已作为当前证据层）

本节保留 2026-06 中旬的阶段性判断。该判断已经完成其历史任务：它把项目从单纯调 backbone 或直接 late fusion，推进到 RGB 过拟合多因素审计。当前这些审计结果已经被 task-nuisance 主线继承为证据层。历史当时的路线为：

```text
split integrity audit
-> temporal sampling audit / ablation
-> training overfit curve summary
-> alignment geometry audit
-> embedding identity retrieval
-> severity calibration verification
-> task inconsistency mixed-factor audit
-> stable behavior subset
-> RGB + behavior late fusion
```

这一路线更符合论文项目的可解释性、可消融性和长期扩展需求，也能避免把黑边、身份外观、OpenFace 对齐几何、时序采样和标签压缩混成一个笼统原因。

Shortcut Audit 的最小可行版本应优先落地：

```text
OpenFace CSV -> subject-level quality summary
quality summary + predictions.csv -> correlation heatmap
quality summary + residual -> residual dependency report
shortcut_audit_report.md
```

## 2026-06-15 RGB 黑填充与硬边界伪迹假设

第一轮 RGB 输入消融后，当前研究重点从“继续堆叠 behavior / late fusion 任务”调整为“解释 RGB 输入模型为什么过拟合”。这一调整更有论文研究价值，因为它直接面向当前模型失败机制，而不是在未解释 RGB 捷径的情况下叠加更多模块。

### 观察

OpenFace aligned face 并不是完全干净的面部行为输入。样例帧显示：

- 脸部轮廓外由纯黑像素填充；
- 遮挡面部的麦克风等区域也可能呈纯黑块；
- 黑色填充与真实脸部区域之间存在硬像素突变；
- 对 DeiT/ViT 这类 patch-based 模型来说，这类高对比边界可能成为稳定但不可泛化的视觉捷径。

第一轮输入消融支持该怀疑：

- `center_mask` 当前测试表现最好，MAE 约 `7.94`，RMSE 约 `10.16`，CCC 约 `0.48`；
- 原始 `rgb` MAE 约 `8.91`，RMSE 约 `10.95`，CCC 约 `0.29`；
- `boundary_erased` 接近或略优于 `rgb`，但不如 `center_mask`；
- `grayscale` 和 `blur` 变差，说明颜色或高频身份纹理不是唯一主因。

### 新假设

```text
OpenFace aligned face 中的黑填充、硬裁剪边界和黑色遮挡块
  -> 被 RGB backbone 学成非行为捷径
  -> 导致 subject-level 泛化不稳、prediction compression 和 case-level 错误
```

这个假设比“背景过拟合”更准确，因为当前输入已经经过 OpenFace 对齐，真正残留的风险是 aligned crop 自身的处理伪迹。

### 下一轮验证

新增输入变体用于验证黑填充和硬边界机制：

- `black_to_gray`：近黑区域替换为中性灰；
- `black_to_mean`：近黑区域替换为当前帧非黑像素均值；
- `black_to_blur`：近黑区域替换为模糊估计；
- `soft_center_mask`：用软边界替代硬 mask；
- `inner_crop_resize`：裁掉外围黑边后 resize。

新增离线审计：

```text
aligned frames + prediction CSV
-> black_ratio / border_black_ratio / center_black_ratio / black_boundary_edge_ratio
-> correlation with true_bdi / pred_bdi / residual / abs_error
```

### 判读规则

- 如果 `black_to_gray`、`black_to_mean` 或 `black_to_blur` 明显优于 `rgb`，说明黑填充本身就是重要捷径。
- 如果 `soft_center_mask` 优于 `center_mask`，说明原先 mask 的硬边界仍在制造新伪迹，后续应使用软 mask 或更自然的图像修复。
- 如果 `inner_crop_resize` 明显改善，说明外围边界/黑边比中心行为区域更影响泛化。
- 如果所有黑伪迹变体都不能改善，但 `center_mask` 仍最好，应检查中心区域选择、面部行为区域和 attention/occlusion 之间的关系。
- 如果 severe 低估仍不改善，应把它作为独立问题继续分析，包括标签分布、loss/sampling、subject bias 和 depression severity 表达差异。

论文表述建议：

> OpenFace alignment removes much of the raw background, but it may introduce structured preprocessing artifacts such as black padding, hard crop boundaries, and black occlusion regions. These artifacts can become shortcuts for patch-based visual backbones. We therefore diagnose and ablate alignment artifacts before introducing additional behavior-fusion modules.

### 诊断后的修正

黑伪迹审计已完整匹配 100 个测试视频，结果显示黑区在 OpenFace aligned face 中非常普遍，但不是强线性解释变量。最大绝对相关约 `0.207`，说明不能把 RGB 过拟合简单归因于单一黑像素指标。

更合理的解释是：

- 边界黑区主要对应 OpenFace 对齐和裁剪填充，更适合作为 artifact 指标；
- 中心近黑像素语义混杂，可能是鼻孔、嘴角阴影、胡须、自然面部阴影、麦克风遮挡或其他真实遮挡，不能直接视作预处理伪迹；
- 高边界黑区四分位的测试误差明显高于低边界黑区四分位，约 `12.29` vs `7.45`，提示边界黑区是泛化风险因子；
- 在 moderate/severe 样本中，边界黑区越多，预测越容易偏低，但样本量较小，应作为 case study 和后续消融的线索；
- `black_to_gray` 优于 `rgb` 但弱于 `center_mask`，说明黑像素替换有帮助，但外围非行为区域、脸部轮廓、裁剪形状、尺度和姿态残留也可能共同造成过拟合。

下一步不再粗暴替换全部近黑区域，而应只处理与图像边界连通的黑区：

```text
border_black_to_gray
border_black_feather
center_mask_black_to_gray
```

这些变体的目标是验证：保留鼻孔、嘴部阴影、麦克风等中心黑区语义信息时，仅中和 OpenFace 边界填充是否还能改善泛化。

## 2026-06-15 RGB 过拟合因素地图

黑填充和硬边界伪迹只是 RGB 输入过拟合的一部分。结合当前实验、OpenFace aligned face 的数据形态、行为 baseline 结果和相关研究，后续应把 RGB 过拟合拆成以下可验证因素，而不是继续寻找单一原因。

### 1. 身份与静态外观捷径

人脸 RGB 图像天然包含强身份信息，包括脸型、年龄、性别、肤色、皱纹、眼袋、胡须、发际线、眼镜、皮肤纹理和面部胖瘦。小样本 subject-independent 任务中，视觉 backbone 很容易学习这些稳定外观，而不是学习跨 subject 的面部行为动态。
眼镜、麦克风、胡须等因素尤其值得作为子问题审计。它们一方面是身份/外观线索，另一方面也是局部遮挡或局部高对比 artifact：眼镜会带来镜框边缘和反光，麦克风可能形成口部附近黑色遮挡块，胡须会改变下半脸纹理和嘴部边界。这些线索不一定与抑郁有因果关系，但在 AVEC2014 这类小样本视频任务中，可能与 subject、任务录制条件、说话方式或 severe/minimal 分布偶然共现，从而成为 shortcut。

相关研究线索：

- Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780
- IDEnNet / Identity-Enhanced Network for Facial Expression Recognition: https://arxiv.org/abs/1812.04207
- FacialPulse: https://arxiv.org/abs/2408.03499

本项目证据：

- `center_mask` 优于 `rgb`，说明去除外围区域和部分静态外观后泛化改善；
- `blur` 变差，说明不是所有细节都是坏信息，模型仍需要局部面部线索；
- behavior-only full OpenFace 特征强过拟合，说明 raw landmark 和 static geometry 也可能携带身份信息。

建议验证：

- 设计 `face_contour_erased`、`eye_mouth_only`、`upper_face_only`、`lower_face_only` 等区域消融；
- 增加 `glasses_region_erased`、`mouth_occluder_erased`、`beard_lower_face_erased` 或手工标注 case study，用于验证眼镜、麦克风和胡须区域是否驱动预测；
- 检查 embedding 是否按 subject、脸型或外观聚类，而不是按 BDI 聚类；
- 训练 subject/identity proxy classifier，测试当前 RGB embedding 是否容易预测 subject。

### 2. OpenFace 对齐几何与裁剪伪迹

除了黑边，OpenFace aligned face 还可能保留或引入脸部尺度、裁剪位置、眼距、脸部中心偏移、插值模糊、头发/衣领/麦克风残留和轮廓形状等几何线索。这些变量可能和数据采集条件、subject 或任务相关。

最新坐标尺度确认显示：当前 OpenFace CSV 中的 landmark `x_*` / `y_*` 不是模型输入的 112x112 aligned face 坐标，而是 OpenFace 原始检测坐标系。示例 CSV 的 `x` 范围约 `150-643`、`y` 范围约 `-11-582`，而实际输入 jpg 为 `112 x 112`。OpenFace 日志中的 camera parameters `500,500,320,240` 提示源坐标系约为 `640 x 480`，因此 geometry audit 已按该尺度重跑。该方向更准确的表述应是 **pre-alignment detection geometry confound**：原始检测阶段的 face scale、bbox、eye distance 和 landmark jitter 可能通过后续裁剪、对齐、缩放、黑边填充和插值过程间接影响 112x112 RGB 输入。

建议验证：

- 从 landmark 外接框统计 face scale、face center offset、eye distance、bbox aspect ratio；
- 统计 aligned 后脸部是否偏上、偏下、偏左或偏右；
- 将这些几何量与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关性；
- 对 high-error case 检查是否同时存在异常尺度、偏移或裁剪残留。

当前 C 任务结果：

- `300` 个 OpenFace CSV 视频汇总，`100/100` 个 test prediction row 成功匹配；
- 最大绝对相关约 `0.3746`；
- `landmark_bbox_height_mean` 与 `true_bdi` 相关约 `r = 0.375`；
- `landmark_bbox_area_mean` 与 `true_bdi` 相关约 `r = 0.341`；
- `landmark_bbox_width_mean` 与 `true_bdi` 相关约 `r = 0.304`；
- `eye_distance_mean` 与 `true_bdi` 相关约 `r = 0.299`；
- `landmark_bbox_height_mean` 与 `residual` 相关约 `r = -0.260`，提示较大的检测 bbox 与高 BDI 样本低估有关。
- 使用 `640 x 480` 源坐标尺度重跑后，`normalized_face_scale_mean` 约在 `0.30-0.40`，并与 `true_bdi` 相关约 `r = 0.341`。

论文表述边界：

- 可以说 OpenFace 原始检测几何与 BDI/severity 和 residual 存在中等相关，是 RGB shortcut 的候选混杂因素；
- 不应说模型直接看到了更大的 112x112 landmark scale，因为模型输入帧已经 resize 到 112x112；
- `normalized_face_scale_mean` 和 `face_center_offset_*` 在使用 `640 x 480` 源坐标尺度后可作为相对检测几何指标解释，但仍不能表述为模型直接观察到的 112x112 坐标。

### 3. 姿态、gaze 与追踪质量捷径

OpenFace 的 `confidence`、`success`、pose、gaze、landmark jitter 既可能是真实行为线索，也可能是追踪质量或采集条件混杂变量。抑郁相关行为可能包括低头、凝视减少和面部动作减少，但模型也可能只是学到了低质量追踪或特定姿态。

相关研究线索：

- OpenFace / OpenFace 3.0: https://arxiv.org/abs/2506.02891
- LibreFace: https://arxiv.org/abs/2308.10713
- Action Unit depression biomarkers: https://arxiv.org/abs/2407.13753

建议验证：

- 扩展 Shortcut Audit，加入 pose/gaze mean/std、confidence mean/std、failed frame ratio 和 landmark jitter；
- 使用 grouped-CV shortcut-only predictor 判断质量/姿态变量能否预测误差；
- 对 severe 低估 case 检查是否伴随低 confidence、大姿态或高 jitter。

### 4. 视频长度、采样与 padding 捷径

黑伪迹审计中最高相关项是 `frame_count` / `sampled_frame_count` 与 `pred_bdi`，约 `r = -0.207`。这提示模型预测可能受视频长度、有效帧数、采样覆盖或 padding/mask 影响。

可能机制：

- 不同任务视频长度不同；
- 长视频包含更多中性片段，稀释抑郁相关行为；
- `MAX_SEQ_LEN` 截断导致长视频关键片段丢失；
- chunk sampling 或均匀采样方式影响关键表情/动作覆盖；
- 有效帧比例影响 temporal pooling。

建议验证：

- 统计 `frame_count`、`sampled_frame_count`、`valid_ratio` 与预测和误差的相关性；
- 对比固定帧数均匀采样，例如 256 / 512 / 1024；
- 对比 first / middle / uniform / random temporal crop；
- 用 temporal occlusion 检查模型是否依赖少数片段。

当前 RGB prediction 审计结果已经支持把该项保留为 P0 混杂因素：`100/100` 个 test prediction row 成功匹配，最大绝对相关约 `0.220`。其中 `truncated_frame_count` 与 `pred_bdi` 相关约 `r = -0.220`，`truncated_ratio` 与 `pred_bdi` 相关约 `r = -0.209`，`frame_count` / `sampled_frame_count` 与 `pred_bdi` 相关约 `r = -0.207`。长视频四分位预测更低、误差更高且无 padding，说明问题更可能来自首段采样、长视频截断或关键片段覆盖不足，而不是简单 padding。该证据弱于 alignment geometry，但足以支撑 fixed uniform 与 temporal crop 消融。

### 5. Freeform/Northwind 任务语境差异

同一 subject 在 Freeform 和 Northwind 中可能表现出不同的说话内容、眼神方向、头部运动、朗读节奏、表情强度、遮挡和视频长度。若模型对任务语境敏感，它学到的可能是 task-specific visual pattern，而不是稳定 depression trait。

建议验证：

- 分任务报告 MAE/RMSE/Pearson/CCC；
- 对同一 subject 的 Freeform/Northwind 预测差异排名；
- 检查 task inconsistency 是否与视频长度、pose、gaze、黑边或追踪质量相关；
- task name 只作为诊断变量，不应在当前阶段直接作为训练输入。

### 6. 标签分布与 prediction compression

当前 RGB 模型存在明显预测压缩：真实 BDI 标准差约 `11.5`，而 `rgb` 预测标准差约 `6.1`。这会导致 minimal 高估、severe 低估。该问题不一定由输入伪迹造成，也可能来自 MSE/MAE 在小样本长尾标签分布下向均值收缩。

建议验证：

- 每个实验都报告 prediction mean/std 与 true mean/std；
- 做 severity group calibration；
- 尝试 severity-balanced sampler 或 label-bin balanced sampler；
- 尝试加权 MSE / Huber / CCC loss，但必须和输入捷径诊断分开做。

### 7. ViT/DeiT patch 级捷径

DeiT/ViT 这类 patch-based backbone 可能对局部高对比 patch、边界 patch、麦克风黑块、眼镜反光和裁剪残留非常敏感。黑边的全局相关性弱，不代表 patch-level attention 中没有局部捷径。

建议验证：

- 使用 attention rollout、Grad-CAM 或 input-gradient 诊断；
- 做 patch occlusion sensitivity；
- 分区域统计 eye、brow、mouth、lower face、face contour、boundary patch 的遮挡影响；
- 将高误差和低误差 case 都纳入可视化，避免只看失败样本。

### 8. 数据增强与鲁棒性不足

如果训练增强不能覆盖 test 中的 crop、黑边、光照、姿态和尺度变化，模型会记住固定预处理风格。后续增强应面向 artifact robustness，而不是一次性加入强增强。

建议优先级：

1. border fill randomization；
2. small affine jitter；
3. mild brightness/contrast jitter；
4. temporal random crop 或 uniform sampling；
5. 只在完成诊断后，再考虑更强的数据增强组合。

### 推荐排查顺序

```text
border-connected black/crop artifact
-> frame_count / temporal sampling audit
-> face scale / alignment offset / landmark bbox audit
-> identity/static appearance audit
-> task inconsistency audit
-> severity calibration and prediction compression
-> targeted robustness augmentation
```

当前研究表述应避免说“RGB 过拟合由黑边导致”。更准确的表述是：

> RGB 过拟合可能来自身份静态外观、OpenFace 对齐几何、边界填充、姿态/追踪质量、视频长度/采样、任务语境差异和标签分布压缩的共同作用。黑边是可见且可操作的风险入口，但不是单一充分解释。

## 2026-06-14 Behavior-only baseline 结果后的研究路线修订

最新 behavior-only baseline 使用 OpenFace 结构化特征进行 BDI 回归，结果显示训练集拟合很强但泛化不足：test MAE 约 `9.93`，RMSE 约 `12.86`，CCC 约 `0.151`；best validation RMSE 约 `12.38`，但对应 train RMSE 只有约 `2.74`。这说明 OpenFace 行为表征路线仍然有研究价值，但不能直接把“所有 OpenFace 特征”视为可靠行为表征。

新的研究假设：

- AU、landmark motion、pose/gaze motion 可能比 raw landmark 坐标更接近可泛化的行为动态；
- raw landmark coordinates 和静态 facial geometry 可能携带较强身份线索；
- OpenFace quality、confidence、success 和 tracking stability 既可作为质量控制变量，也可能成为预测捷径；
- behavior-only baseline 弱于 RGB 不代表行为线索无效，可能是特征组混杂、模型容量过大或评价粒度不够导致。

下一阶段实验应优先回答三个问题：

1. 哪些 OpenFace 特征组可以在 subject-level 泛化中稳定降低误差？
2. RGB/MTL-Lite 与 behavior-only 的错误样本是否重叠，还是互补？
3. severe 低估、minimal 高估和 Freeform/Northwind 不一致是否能被某些行为特征组解释或缓解？

因此，行为路线的实验顺序调整为：

```text
behavior prediction export
-> feature-group ablation
-> RGB vs behavior case overlap
-> stable behavior subset
-> late fusion
-> behavior auxiliary MTL
```

论文表述上，当前 behavior-only baseline 可作为一个重要诊断结论：直接使用完整 OpenFace CSV 特征并不会自动获得可泛化抑郁表征，必须通过特征组消融、去身份化和行为动态约束来筛选可靠线索。

## 2026-06-15 高优先级过拟合验证路线

权威路线文档：`docs/RGB_OVERFITTING_AUDIT_PLAN.md`。本节记录研究动机与论文表述，具体执行清单以 `docs/TODO.md` 为准，具体输出规格以 `docs/SHORTCUT_AUDIT_DESIGN.md` 为准。

在完成第一轮 RGB 输入消融、黑伪迹审计、边界连通黑区消融和统一 prediction summary 后，当前研究重点应从“继续尝试更多输入 mask”转向“验证 RGB 过拟合的关键机制”。原因是现有结果已经说明：

- `center_mask_black_to_gray`、`center_mask`、`border_black_feather` 均能不同程度改善原始 `rgb`；
- 但这些变体仍未解决 prediction compression；
- `center_mask_black_to_gray` 虽然整体 MAE 最好，却加重 severe 低估；
- `gray_scale`、`blur`、`inner_crop_resize`、`black_to_mean` 不支持颜色、纹理或外围区域任一单因素解释；
- 因此继续做相似 mask 的边际论文价值下降。

更值得优先验证的高价值问题如下。

### 1. Split / subject integrity

所有 subject-independent 泛化结论都依赖 split 正确性。必须确认：

- train/val/test 是否 subject-disjoint；
- 同一 subject 的 Freeform/Northwind 是否被放入同一 split；
- video_id 规范化是否导致误匹配；
- 是否存在重复视频目录、重复标签、标签文件与视频目录前缀不一致。

该审计不直接证明过拟合来源，但它是所有后续结论的有效性前提。

### 2. Temporal sampling and sequence coverage

当前黑伪迹审计中，`frame_count` / `sampled_frame_count` 与 `pred_bdi` 的相关性最强，提示模型可能对视频长度、采样位置、截断和有效帧比例敏感。高优先级验证包括：

- 真实数据上运行 temporal sampling audit；
- 对比 `uniform_256`、`uniform_512`、`uniform_1024`；
- 对比 `first_crop`、`middle_crop`、`random_crop`；
- 将每组结果接入 prediction run summary，报告 prediction std、severity bias 和 task consistency。

如果固定帧数采样显著缓解 prediction compression 或 task inconsistency，则当前均值池化/截断策略可能是主要过拟合入口之一。

### 3. Training curve overfit gap

仅看 test MAE 不足以解释过拟合。应跨实验统计：

- best validation epoch；
- train RMSE / MAE 与 val RMSE / MAE 的 gap；
- val 最优后 train 是否继续下降但 val/test 不改善；
- behavior baseline、RGB baseline 和各输入消融的 gap 是否一致。

该分析可以区分两类情况：输入变体真正改善泛化，或只是改变预测分布和测试集偏置。

### 4. OpenFace alignment geometry

OpenFace aligned face 不只包含黑边，还包含 face scale、bbox shape、face center offset、eye distance、轮廓形状和插值痕迹。建议把 alignment geometry audit 提升到 P0：

- 从 OpenFace landmark 坐标计算 bbox area/width/height/aspect；
- 计算 face center offset、eye distance、normalized face scale；
- 与 `true_bdi`、`pred_bdi`、`residual`、`abs_error` 做相关；
- 与 severe 低估、minimal 高估和 task inconsistency 做分组比较。

如果几何变量能解释误差或任务不一致，则 RGB 模型可能利用了对齐几何和静态人脸尺度，而不是抑郁相关行为。

### 5. Embedding identity retrieval

身份信息不建议首先用 subject classifier 验证，因为 subject-independent split 中 test subject 未在 train 出现。更适合先做 paired-task retrieval：

- 提取同一模型的 test embedding；
- 对每个 Freeform 样本查找最近邻，看 Northwind paired sample 是否排在 top-k；
- 反向从 Northwind 查 Freeform；
- 同时比较 embedding 是否按 BDI severity 聚类。

如果同 subject 的两个任务 embedding 高度互为近邻，而 severity 聚类弱，说明 RGB backbone 更强地编码了身份/静态外观。

### 6. Severity calibration verification

当前所有较好输入变体仍存在 `pred_std < true_std`。应使用 val set 拟合简单 post-hoc calibration，并只在 test 上评估：

```text
pred_calibrated = a * pred + b
```

目的不是调出最终模型，而是验证 severe 低估和 minimal 高估是否主要来自 prediction compression。如果 calibration 明显改善 severe/minimal 但不改变排序相关性，则后续应单独研究 severity-aware sampler、weighted loss、Huber/CCC loss。

### 7. Task inconsistency mixed-factor audit

Freeform/Northwind 同 subject prediction diff 需要与多因素关联：

- frame_count / sampled_frame_count；
- black_border_ratio；
- confidence / success；
- pose / gaze；
- alignment geometry；
- prediction severity bias。

如果 task inconsistency 能被这些变量解释，论文中应把任务语境作为混杂因素，而不是把两个任务简单视作同分布重复样本。

### 更新后的论文主线

推荐将论文实验路线组织为：

```text
1. Baseline failure: prediction compression, severe underestimate, task inconsistency
2. RGB input artifact ablation: center mask and border artifact evidence
3. Multi-factor shortcut audit: temporal, geometry, identity, quality, task context
4. Calibration analysis: separate shortcut mitigation from severity compression
5. Behavior feature baseline: evaluate whether structured facial dynamics generalize
```

核心表述应保持克制：

> 黑边和硬边界伪迹是 RGB 过拟合的重要可见入口，但不是唯一原因。当前证据更支持一个多因素解释：RGB 模型同时受到 OpenFace 对齐几何、身份静态外观、时序采样、任务语境和标签分布压缩的影响。

## Identity Suppression 与干扰因素消除方法谱系

本节用于归档现有研究中常见的身份记忆、静态外观和输入 artifact 抑制方法，并映射到当前 OpenFace aligned face / AVEC2014 小样本场景。

### 1. 输入级去身份化与外观弱化

常见方法包括 face contour / hairline masking、skin texture smoothing、颜色/风格随机化、局部饰物遮挡、只保留眼口/AU 相关区域、face anonymization / de-identification。该类方法的优点是可解释、实现成本低、适合小样本；缺点是可能同时损伤表情/AU/行为线索。

对当前项目最匹配。已有结果显示：`center_mask` 最健康，`border_black_feather` 改善 severe bias 但增强 identity retrieval，`middle_crop` 降低 identity retrieval 但损害 task consistency。因此输入级身份弱化应先以机制消融方式验证，而不是直接替换主模型。

### 2. 表征级 adversarial identity removal

Domain-adversarial / gradient reversal 方法通过让共享表征对主任务有效、但对 domain / subject 分类器不可辨别，学习 domain-invariant representation。参考：Domain-Adversarial Training of Neural Networks, https://arxiv.org/abs/1505.07818。

映射到本项目，可以把 `subject_id` 作为 adversarial target：

```text
video embedding -> BDI regression / severity ordinal
video embedding -> Gradient Reversal -> subject classifier
```

但该方向应放在输入级机制消融之后。原因是 AVEC2014 小样本、subject 数多、每个 subject 的任务视频有限，subject-adversarial head 容易训练不稳定，也可能把与抑郁严重程度纠缠的有效行为信号一并抹除。

### 3. Disentanglement：身份 / 姿态 / 表情解耦

FER 和 face representation 研究中常见的路线是把 identity、pose、expression 或 behavior 表征显式拆开，或学习 identity-distilled / identity-dispelled features。该方向可以为论文讨论提供理论支撑，但直接实现完整 disentanglement 或生成式去身份化对当前项目风险较高：数据量小、训练复杂、生成伪影难以解释。

当前更适合做轻量替代：输入级身份纹理抑制、embedding identity retrieval 诊断、case-level spatial occlusion，再决定是否进入 adversarial / disentangled representation。

### 4. 行为表征替代 RGB 外观

抑郁视觉分析更理想的信号应来自 AU、landmark motion、pose/gaze dynamics、面部活动迟滞等行为动态，而不是静态 RGB 外观。AU temporal biomarker 和 facial dynamics 相关研究支持将结构化 OpenFace features 作为行为对照。参考：Exploring Facial Biomarkers for Depression through Temporal Analysis of Action Units, https://arxiv.org/abs/2407.13753。

这与当前 behavior-only baseline 路线一致。长期目标不是把 RGB 修补到完全可靠，而是判断 RGB、behavior features 和二者融合各自是否捕捉到可泛化的 severity signal。

### 5. 与 shortcut learning 的关系

Shortcut learning 研究指出，深度模型可能利用训练分布中容易但非因果的线索，而不是目标任务真正需要的机制。参考：Shortcut Learning in Deep Neural Networks, https://arxiv.org/abs/2004.07780。当前项目中的身份纹理、脸部轮廓、眼镜/胡须/麦克风、OpenFace 黑边硬突变、对齐几何、视频长度和任务语境都符合 shortcut 风险特征。

### 当前项目方法匹配度

| 方法 | 匹配度 | 当前判断 |
|---|---:|---|
| `edge_soften_only` / `border_blur_fill` | 高 | 直接检验 OpenFace boundary hard-transition artifact |
| `identity_texture_suppressed` | 高 | 针对高频皮肤纹理、胡须、发际线、眼镜反光等静态身份线索 |
| face contour / hairline weakening | 高 | 弱化脸型、发际线、外轮廓 identity cue |
| local accessory occlusion case study | 高 | 验证眼镜、麦克风、胡须等局部 shortcut |
| style / color augmentation | 中高 | 可弱化颜色/纹理 shortcut，但必须避免损害行为线索 |
| subject-adversarial GRL | 中 | 有理论支撑，但小样本不稳定，放在输入级消融之后 |
| full generative de-identification | 低 | 复杂且可能引入生成伪影，暂不作为当前主线 |
| severity-aware sampler / loss | 中 | 必要，但应在 identity / boundary 机制拆清之后测试 |

### 让模型回到正轨的判定标准

一个方法不能只凭 MAE 下降被视为有效。更健康的方向应同时满足：

```text
same_subject_top1/top5 下降或不升高
severity_agree 不下降，最好上升
CCC 不下降
pred_std 不继续压缩
severe bias 改善
minimal bias 不明显恶化
task_diff_mean 不恶化
```

若 identity retrieval 下降但 severity agreement 和 task consistency 也下降，应解释为 `middle_crop` 式失败。若 severe bias 改善但 identity retrieval 上升，应解释为 `border_black_feather` 式 identity / artifact 纠缠。
