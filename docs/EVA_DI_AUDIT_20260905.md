# EVA-DI 逐行审计 + 端到端验证报告（2026-09-05）

包范围：对 `src/eva_di/**`（20 py）、`configs/eva_di/**`（9 yaml）、`tests/test_eva_di_*`（14 文件）做逐行审计；实施修复；按已授权的 E2E 冒烟边界运行（写路径仅 `/tmp/eva_di_e2e/`，GPU 4060，无 commit）。生产缓存根 `/home/zhen/dataset/depression/avec/2014/eva02_features_v1` 未触碰；全量抽取 / oracle 等价 / 训练矩阵 / commit 仍为独立未授权包。

---

## 1. 已实施修复（问题 → 根因 → 修法）

| # | 模块:位置 | 问题（严重度） | 修复 |
|---|---|---|---|
| 1 | `extract/encoder.py` 构造尾部 | **真实 timm 下必崩**：`cfg["input_size"][0]` 取的是 timm `(C,H,W)` 的通道数 3 而非边长，断言永不通过（单元层不可见，E2E 第 1 次运行抓到） | 取 `tuple(cfg_input)[-1]`，报错信息打印完整 cfg 元组 |
| 2 | `extract/runner.py:extract_video` | **真实数据必崩**：`aligned_image_path` 以 `of3_root/samples/<sid>` 为基拼接，但 OF3 v2 真实 csv 该列已是 of3_root 相对路径（`samples/<sid>/aligned/...`）→ 路径翻倍；合成 fixture 用的是 sample-dir 相对语义，与真实数据不一致导致单测失明 | 统一以 `paths.of3_root` 为基（含 SHA 校验与解码两处）；`tests/_eva_di_synth.py` 列值同步为真实语义；registry 注释钉死"相对值 of3_root-相对" |
| 3 | `train.py:run_training` | **结果性错误**：severity 权重表按 `sev_table_np[:len(records)]` 位置切片，与 `train_records` 顺序无对应保证 → 权重错配 | `weight_by_id` 字典按 `sample_id` 取权重 |
| 4 | `train.py:derange` | **语义破坏**：按 subject 建 dict 再 zip，同 subject 多条录音被合并、置换静默失真 | 位置式 `dataclasses.replace(r, subject=int(s)) for r,s in zip(train_records, permuted)` |
| 5 | `train.py` | 静默 CPU 回退（3 处：train/encoder/export）：`device='cuda'` 无 CUDA 时悄悄落 CPU，污染 provenance 的 device 字段与吞吐预期 | 统一 fail-fast：`cuda` 请求而 CUDA 不可用即抛 `EvaDiError`，要求显式 `device='cpu'` |
| 6 | `train.py:_evaluate` | 指标未按 Stage C 契约 clamp 到 BDI 量程 | `y_hat` clamp [0,63] |
| 7 | `train.py` | fp16 GradScaler 默认语义与 bf16-mixed 不符 | `torch.amp.GradScaler("cuda", enabled=False)` 常关，保留 unscale/clip/step 序列 |
| 8 | `train.py` | 泄漏守卫依赖缓存与 `max_recordings`（子采样后守卫可被削弱） | 常开 registry 级守卫：对**全量 train split** 的 csv 重算并断言与 pinned train-only 统计一致；结论写入 `val_metrics.json`/summary |
| 9 | `train.py` | best.pt 落在 run_dir 根、无 best 记录、无独立 val 指标文件 | `run_dir/checkpoints/best.pt` + `best_record` + `val_metrics.json` |
| 10 | `train.py` | run_id 为 null 时不可复现、resolved 配置未落盘 | run_id 回退 `{yyyymmdd}-{config_name}-seed{seed}`；写 `config_resolved.yaml` |
| 11 | `export_embeddings.py` | **split='val' 直接崩溃**（单独建索引时 subject 表要求 train）；且全量索引建两遍 | 单次 `build_recording_index(paths, splits=("train", split))`，导出消费 `by_split(split)`；IO 减半 |
| 12 | `export_embeddings.py` | 输出命名/模式未定 | `run_dir/embeddings/p_mean_<split>.npz` + `v_h0_<split>.npz`（schema/split/sample_ids/embeddings），返回 `list[Path]` |
| 13 | `extract/cache_writer.py`+`runner.py`+`contracts.py`+`cache_reader.py` | manifest 缺 of3 根与 csv 指纹 → 缓存无法自证来源；reader 无法拒收错根缓存 | manifest 新增 `of3_root`、`features_csv_sha256_by_sample`（REQUIRED_MANIFEST_FIELDS 强制）；entry 记 `features_sha256`；训练入口对 manifest 做 pinned 指纹断言 |
| 14 | `extract/runner.py` | 无显式 worklist（冒烟只能全量或前 10）；日志无吞吐/指纹 | `samples=` 显式 worklist（须为 universe 子集，覆盖 smoke）；`jobs_sha256`、`throughput_frames_per_s`、逐视频 `seconds`；`main()` 拒绝 `run.mode != 'extract'` |
| 15 | `config.py` | `data.behavior_norm` 键缺失（v1 需钉死 reuse） | 新增键，校验 ∈{reuse,recompute}；`config_name`/`resolved_payload` 进 provenance |
| 16 | `extract/runner.py` 黑帧守卫 | **P0 静默特征损毁**：实测本环境 torch 2.8.0+cu129 上 `uint8_tensor.any(dim=1)` 返回 **uint8** 而非 bool；uint8 "mask" 使 `frames_uint8[nonblack]` 退化为整数花式索引（全 1 → 同一帧重复编码），`full_cls[keep] = cls.numpy()` 把 512 行特征**塌缩到第 1 行**（E2E 落盘缓存实测：cls.npy 512 行仅 1 行非零，frame_valid 却全 True）。任何代码评审都无法预见，唯 E2E+落盘检查可抓 | `amax(dim=1) > 0`（比较运算恒 bool，版本安全）+ `keep.astype(bool)` 兜底；新增 4 个钉死测试（身份映射/比率守卫/blank 行/部分残缺致命） |
| 17 | `of3_registry.py` | **真实数据拒收**：train 3 个视频共 27 行"对齐失败行"（`image_valid=False` 且 `aligned_mask_schema=''`、path/sha/landmarks 全空、status=`not_run_*`），全行严格 schema 断言直接崩掉训练入口（泄漏守卫加载全量 train 时首触）| registry 语义钉死：`image_valid=False` 且 schema 为空 → 合法掩码行（零填充，不解析行为列）；schema 非空仍须等于钉定值（半残状态依旧致命）；fixture 增加 `blank_schema_rows` 同构覆盖 |
| 18 | `of3_registry.py:_parse_behavior_row` | **P0 特征空间错配**（E2E 泄漏守卫抓到，delta mean 104.7）：pinned `technical.json` 的 landmark 轴统计定义在归一化空间 `2*v/111-1`（轴名 `*_norm` 即证据；权威实现在 trans `ribformer/data/clips.py:851`），registry 却存原始 112-px 像素 → `normalize()` 产出 ~150 量级垃圾，**模型全部行为特征损坏**。synth fixture 用同一错误规则自洽生成 → 单测失明 | registry 输出改为钉定 `2*v/111-1`（float32 数组 ufunc 路径，与生成器逐位同序）；synth 侧同步；新增钉死测试（含逐位一致断言）；`behavior.py` 容差按真实数据校准（abs 1e-3/rel 5e-2，见 §4-歧义5）|
| 19 | `train.py` checkpoints | E2E 实锤 §4-歧义4：val=2 且预测恒定 → `val_ccc=NaN`（CCC 数学上未定义）→ best.pt 永不落盘 → 导出环节无产物可用 | 对齐 trans B0 参考（`behavior_b0.py:679` 每 epoch 存 last.pt）：`checkpoints/last.pt` 恒存；summary 增 `checkpoint` 字段（best 优先，NaN 情形回退 last）；选择指标不变 |
| 20 | `export_embeddings.py` | **导出必崩**（E2E 首触）：identity 头是 lazy+train-gated，导出侧重建模型时头未实例化 → 训练 checkpoint 的 `frame_head/video_head` 权重成 unexpected keys，strict 加载 RuntimeError | 导出按训练同构构造（identity flags + subject 数）并新增 `DualStreamDI.materialize_identity_heads()` 先建头再 strict load；`load_state_dict` 保持 strict=True（结构漂移仍会立刻暴露）；钉死测试含 strict 失败→修复成功对照 |

E2E 前置存量修复（本审计周期前半段完成，此处并入台账）：`entry_matches` 键名错误（resume 永不命中）、`collate` 静默修补 ragged mask、`ce_masked`/masked-mean 广播崩溃、`paths.py` 缺 `EVA_WEIGHT_REVISION` 导入、`model.identity_logits` 设备引用、`by_split('test')` 裸 KeyError、masking 测试语义改写、`_reg_to_record` 死代码删除、`tests` 导入约定（`from tests._eva_di_synth import ...`）。

## 2. 性能与阻塞项观察（量化，未改动）

| 项 | 量化 | 判断 |
|---|---|---|
| PIL 单线程解码 112² jpg | ~1–2 ms/帧 → 全量 300 视频 × ≤512 帧 ≈ 153.6k 帧 ≈ 2.5–5 min | 预算内；若嫌慢可进程池并行，非阻塞 |
| `verify_image_sha=True`（默认） | 153.6k 次 SHA-256（每帧两读：校验+解码） | 约 +1 分钟级；安全性优先，保留默认 |
| manifest 每视频整体重写 | ≤300 次 | kill-safe 设计，忽略 |
| 训练启动时的全量 train-csv 一致性重算 | ~150 个 features.csv（约 1 GB IO，秒~几十秒） | 可接受；未来可加缓存指纹 |
| export 逐录音 batch=1 | 30/120 录音规模毫秒~秒级 | 非阻塞 |
| 4060 8GB + bf16 chunk=128@224² | E2E 实测无 OOM | 全量提取同参数安全 |
| λ warmup 双施加 | 梯度反传系数 ∝ scale²（见 §4-歧义1） | 计划歧义，需决策，不改代码 |

## 3. 代码 ↔ 设计文档一致性核对（偏差清单，docs-sync 时回写文档）

| 设计文档 | 代码实际 | 处置建议 |
|---|---|---|
| `load_train_stats` | `load_pinned_stats(root)` | 文档改名 |
| `build_subject_table(records)` | 接受 list，语义同 | 文档补签名 |
| `DIBatch.frames_p_in` | `DIBatch.visual`（另含 `bdi_raw`） | 文档改名 |
| 帧级 identity logits `[B*Tv,S]` | `[B,T,S]` + `frame_mask` | 文档改形状（等价、免打包） |
| `severity_weighted_l2(pred,target,w[B,T])` | `w[B]` 录音级 | 文档改语义 |
| `load_resident`/`ResidentArrays` | 惰性 `load_sample`+memo | 文档改写（内存更优） |
| `resume: bool` | `resume_entry: dict` | 文档改 |
| `build_recording_index(cfg)` | `(paths, splits, behavior_stats, max_recordings, cache)` | 文档改 |
| 编码器返回 ndarray | CPU torch tensor | 文档改 |
| `uniform_select` 无条件 `len>=budget//2` 断言 | 仅降采样分支断言（n<budget 时前者不可能成立） | 文档改（设计原文不可实现） |
| weight_path=`artifacts/…(解包后)` | 默认 HF snapshot 路径（同 sha） | 文档改 |
| `@pytest.mark.eva_di_data` | 未实现（当前无需真实数据的测试） | 文档删或留待 EXTRACT-v1 |
| 验收命令 `import eva_di` | `import src.eva_di` | 文档改 |
| 设计 §6 配置键集 | 新增 `data.max_recordings` | 文档补键 |
| 验证计划授权表 | 仍标 CODE/EXTRACT/SMOKE 未授权（滞后于当前 E2E 授权） | docs-sync 更新 |

## 4. 计划歧义（需用户决策，代码按字面实现并注记）

1. **λ 双重 warmup → scale²**：DUAL 计划 L53"两 λ 共享同一 warmup 调度"且验证计划 Stage C `w_k(e)=W_k·min(1,e/W)`。代码同时字面执行两者（每 epoch `set_lambda(λ·scale)` 且 CE 权重乘 scale）→ 反传净系数 ∝ scale²。单 warmup 变体 = 删除 `set_lambda` 调用（一行）。
2. `warmup_epochs` 一律取 `t1.warmup_epochs`；t3-only 配置下 t3 的 warmup 键被忽略（矩阵内两者恒 5，当前等价）。
3. `behavior_norm: recompute` 在 v1 与 `reuse` 同义（一致性断言代替重算；已写 consistency_note）。
4. best 模型准则未定义 nan 情形：val 有效录音 <2 或预测恒定 → `val_ccc=nan` → best.pt 不保存 → export 断言失败（**已知未修边界**，冒烟 2 条 val 正常触发有限值）。
5. **泄漏守卫"一致性"的可行边界**（真实数据逼出）：pinned 统计由 OF3 quality policy 在 **160494 帧**上拟合，其中 65 帧经 manifest/exclusion registry 剔除，**无法从 features.csv 复原**；csv-only 重算最少 160559 帧，残差实测 mean≤1.3e-4、std-ratio≤1.3e-2。故守卫容差定为 abs 1e-3 / rel 5e-2（仍可拒任何 val/test 混入的 O(σ) 级污染），且**防泄漏本身是结构性的**（统计只在 `split_ids["train"]` 上拟合）。原设计"assert within tolerance"未给数值口径 → 已按实测校准并注记于 `behavior.py`。
6. `behavior_status` 列：trans 侧 `behavior_valid_mask` 还要求 `behavior_status=='ok'`（csv 列），当前 registry 仅用 `behavior_valid ∧ finite`，未交叉校验 status；冒烟数据两者一致，全量差异面待定（未改，列入未闭合）。

## 5. 未闭合项

- oracle 等价 harness 仅有骨架（属 EXTRACT-v1 包：oracle cos≥0.999 证据落 `cache_root/_equivalence`）
- 全量提取运行与生产缓存（未授权）
- `subject_attacker.py` 仍是旧 Stage-B CSV 接口，**npz 消费适配器不存在**（导出→攻击者闭环未通）
- 设计文档偏差回写（§3 清单）+ 验证计划授权表更新（docs-sync 单独包）
- 验证计划中的 `@pytest.mark.eva_di_data` 标记
- 全训练矩阵 / 消融运行（未授权）
- 本批修改**未 commit**（分支 dev @ c867b4b，全部为未跟踪新文件 + 冻结 WIP 不动）

## 6. 计划未明确的细节/操作/边界/数据流（本审计钉死或注记）

- subject 派生：`video_id[:5]`（计划未写）
- behavior 行掩码语义：NaN 位置补 0 + 整行 `behavior_valid=False`（计划未写补值）
- **前缀截断规则**：GRU 无 packing → 首个 behavior-invalid 行后全部丢弃；对 uniform512 最坏损失 ~50% 帧（钉为不变量）
- severity/subject/target/behavior 统计仅 train 维护；val/test 只经缓存读取
- 指标 clamp [0,63]；label smoothing 0.1；GradScaler 常关
- `collate` 逐录音前缀预检 + 复制录音自带 mask（拒绝静默修补）
- resume：粒度=每视频；`entry_matches` 键 `index_sha256`/`valid_sha256`；显式 worklist 覆盖 smoke
- manifest REQUIRED 键扩至含 `of3_root`/`features_csv_sha256_by_sample` → 旧格式缓存被 reader 拒收（当前无存量缓存，安全）
- 设备策略：cuda 请求无 CUDA = 立即失败（3 入口一致）
- 导出布局：`run_dir/embeddings/{p_mean,v_h0}_<split>.npz`；best.pt 在 `run_dir/checkpoints/`
- `aligned_image_path` 相对值 = of3_root-相对（以真实 csv 钉死，fixture 对齐）
- **OF3 v2 对齐失败行状态**（计划未写）：`image_valid=False` 行的 `aligned_mask_schema/aligned_image_path/aligned_image_sha256/landmarks` 可全空（实测 train：227_1 22 行、227_2 2 行、323_1 3 行；status=`failed|not_run_*`）→ registry 掩码而非报错
- **张量掩码纪律**（计划未写，环境逼出）：本环境 `torch.uint8.any(dim)` 返回 uint8；一切掩码必须来自比较运算或显式 astype(bool)（`amax>0` 模式），禁止直接消费 `Tensor.any` 作 mask
- `train` 入口拒绝 test split；导出仅 {train,val}；`main()` 各自模式守卫

## 7. E2E 证据（2026-09-05，全部产物仅 `/tmp/eva_di_e2e/`；`E2E_OK`）

驱动：`/tmp/eva_di_e2e/run_e2e.py`（硬断言 cache/log/output 三根均在 /tmp 边界内；生产缓存根不存在断言）。配置 `e2e_di_full.yaml` extends `di_full.yaml`，`max_recordings=2`、`epochs_max=3`、bf16-mixed、cuda（RTX 4060）。

**提取**（首轮干净缓存）：worklist=train/val 各前 2（`203_1_Freeform/Northwind, 205_1_Freeform/Northwind`），每视频 512 帧、`verify_image_sha=True`、`n_black=0`；吞吐 **428.8 帧/s**（单视频 1.05–1.6s）；`jobs_sha256=0e2936ba…`；逐视频 `features_sha256` 落 manifest。**resume**：二次运行 4/4 `reused`、0.006s、零重编码。
**泄漏守卫**：`passed (full train-split recompute within tolerance)`（160559 csv 可见帧 vs pinned 160494 生成帧）。
**训练**：3 epochs，`train_total 1.327→0.536→0.254`、`train_reg 1.247→0.019`、frame_ce 4.02→3.96、video_ce 3.95→3.87；`val_mae 4.68/0.78/5.93`；`val_ccc=NaN`（val=2 恒定预测 → CCC 未定义，§4-歧义4 现场复现）→ 采用 `last.pt` 回退。
**导出**：`p_mean/v_h0 × train/val` 4 个 npz，shape (2,192)，全有限，mean|.| 0.30–0.41（非塌零）。
**provenance**：`device=cuda`、`config_name=e2e_di_full`、`behavior_norm_sha256=9e9bfee0…`、branch/commit/config 双 sha/manifest sha 齐备；`run_dir=/tmp/eva_di_e2e/outputs/e2e-20260905-1`（config_resolved/run_provenance/metrics.jsonl/val_metrics/summary/checkpoints 全部落盘）。

**E2E 净收益**：抓到 4 个单测不可见缺陷（#16 P0 uint8 掩码特征塌缩、#18 P0 特征空间错配、#20 导出 strict-load 必崩、#19 NaN-ccc 产物缺口）+ 3 个真实数据语义钉死（#2、#17、§4-歧义5）。

## 8. 验证命令（WSL，repo 根）

```bash
# 单元/回归（应 127 passed）
~/miniconda3/envs/light/bin/python -m pytest tests -k eva_di -q
# 全套（应 710 passed / 4 failed / 2 errors；6 项均为冻结 WIP 存量，与无包基线 583 一致：583+127=710）
~/miniconda3/envs/light/bin/python -m pytest tests -q
# 配置加载
~/miniconda3/envs/light/bin/python -c "import sys;sys.path.insert(0,'.');from src.eva_di.config import load_config;from pathlib import Path;[load_config(Path('configs/eva_di')/n) for n in ['eva_di_base.yaml','di_full.yaml','extract_uniform512.yaml']];print('ok')"
# E2E 冒烟（仅 /tmp 根；授权边界内）
~/miniconda3/envs/light/bin/python /tmp/eva_di_e2e/run_e2e.py
```

## 9. 剩余风险

- scale² 若判为偏差，需在训练矩阵前定夺（影响 T1/T3 消融可比性）
- nan-val 边界未修（小验证集实验需 ≥2 条有效 val 录音）
- 攻击者接口适配未做 → 身份度量指标暂不可产出
- E2E 只覆盖 4 视频冒烟；oracle 等价未证前，缓存特征与 trans 基线的数值一致性仍是 EXTRACT-v1 前置门槛

---

## 10. 第二轮全盘逐行审计（2026-09-05，同包优化续）

### 10.1 方法与裁决
9+1 路 subagent 对 `src/eva_di/**` 与 `tests/test_eva_di_*`、`configs/eva_di/*` 全文逐行扫描；所有结论先由主线程对源码逐条核实（grep/read 精确行）后才允许改动。第一批报告存在系统性虚构，典型被否结论（未据此改动）：runner `except ValueError` 吞异常、`or True` 短路、`jobs_sha256` 先于 smoke 计算、`di_full_identity.yaml` 存在、export p_mean 未掩码、registry 接受 inf、绝对 image_path 未处理等——逐一与真实源码比对为伪。后批（data/tests/cache/train+export/model/config/extract-core）逐字准确，采纳前仍复核。

### 10.2 修复清单（级别→根因→修法）
| 级别 | 文件 | 根因 → 修法 |
|---|---|---|
| **P0-A** | `train.py` | 恒等头懒建于 optimizer 参数快照**之后**→头权重在任何配置下都不进优化器→全部 DI 臂无效。改为 `materialize_identity_heads()` 先于 `parameters()`；另加"无可训练参数"守卫。差分测试（1ep vs 2ep `last.pt` 头权重必须不同）作为回归锁 |
| **P1** | `train.py`/`losses.py`/`config.py` | warmup 双重施加：`set_lambda(λ·w)` 且 CE 权重再乘 `w` → 有效强度 ∝ λ·w²（前 4 epoch 0.04/0.16/0.36/0.64 对比预期 0.2/0.4/0.6/0.8）。**语义保持不动、待用户裁决**；config 强制 `t1.warmup_epochs == t3.warmup_epochs`（消除 T3 死键被 T1 键隐式驱动的陷阱）并拒绝 warmup>epochs_max |
| **P1** | `subject_table.py` | DERM 由逐样本位置 shuffle 改为**类空间无不动点双射**（同 subject 恒定映射、频次多重集严格守恒），对齐计划 L77"保持类数/频次"与 GLOBAL_LOCAL 先例"对 subject 做固定无自映射置换"；`n_classes` 参数保证 `max_recordings` 截断下仍置换全类空间。**待用户确认** |
| **P1** | `dataset.py` | `max_recordings` 旧行为同时截断 val → val_ccc 跨 run 不可比。改为**仅截断 train**；val 决策集恒完整（E2E 因此需覆盖全部 100 条 val，已覆盖） |
| **P1** | `runner.py` | ① 既有 manifest 必须先匹配当前环境（schema/backbone/weight sha/revision/policy/budget/timm/torch/of3_root）否则拒写，杜绝两代特征混入同一 manifest；② 复用前实时重算 features.csv sha 对账 entry，漂移→重抽；③ 每视频失败先写 runner log（accepted=false + error）再抛；④ 黑帧守卫移到编码前；⑤ 空视频显式报错；⑥ log 文件名加随机后缀防同秒覆盖 |
| **P1** | `dataset.py` | 跨源断言：csv split 列 vs `dataset_split.json`（OF3 csv 方言 `dev`≙json `val`，白名单 `CSV_SPLIT_ALIASES` 钉死于 contracts）；cache manifest 的 csv-sha 与盘上 csv 逐视频对账 |
| **P0（自纠）** | `export_embeddings.py` | 本人 Round-2 重写时丢失 `.sum(1)`：masked-mean 变逐帧除法，npz 呈 (N,T,D)。回归测试当场捕获后修复，E2E 断言 `emb.shape==(n,192)` 复验 |
| P1 | `export_embeddings.py` | 入口守卫：`run.mode=='train'`、缓存指纹+of3_root 钉、checkpoint provenance 交叉核对（ablation/identity flags/cache manifest sha 不符即拒——消融 forward 不能冒充全模型特征）；npz 附 run/config/checkpoint sha/precision 等 provenance 并原子写 |
| P2 | `config.py` | bool 冒充 int、非有限数、seed<2³²、run_id 路径安全、extends 类型、splits 去重、`weight_decay≥0`、`gru_hidden==proj_dim` 钉死、derange 需启用头、`resolved_sha256`（合并后载荷）与 `source_sha256`（叶文件）分离→base 漂移可见 |
| P2 | `model.py` | 按 flag 建头（state_dict 忠实镜像配置）；masked-mean 零有效帧 raise；gru/proj 不匹配在构造前 raise |
| P2 | `cache_reader.py`/`cache_writer.py`/`contracts.py` | 指纹字段清单钉死且拒绝未知 expected key；entry/meta/manifest 必填字段校验（meta 含 `n_image_valid`）；`features_sha256` 64-hex 校验；meta.json sha+sample_id 复验；tmp 名含 pid + fsync + 目录 fsync；manifest 写入前结构与 entry↔csv-map 交叉校验、损坏 JSON 显式报错 |
| P2 | `paths.py` | 支持 `${VAR}` 展开；`resolve_read_only(mode=...)` 按模式存在性检查（train 不需权重文件、extract 不需标签），文件型 root 校验 `is_file` |
| P2 | `of3_registry.py` | `frame_index_zero` 仅接受纯数字（拒绝 `+5`/`5_0`）；行宽漂移（None 单元/restkey 溢出）即拒；`image_valid` 行空 `aligned_image_path` 即拒（避免下游无上下文的 OSError） |
| P2 | `encoder.py` | `HF_HUB_OFFLINE=1` 硬置（原 setdefault 可被外部 0 值穿透）；uint8 dtype 校验（防双重归一化静默坍缩）；空批返回两个**不同**张量；cuda 缺失 fail-fast；pooling/fc_norm 身份入 metadata |
| P2 | `metrics.py` | `severity_group` 走 float() 转换（np.float32 NaN 曾漏网成 very_severe——正是旧库 bug 复刻）；`_pair` 先比 shape 再 flatten；worst-group 排除 unknown/非有限并以组名序定 tie |
| P2 | `losses.py` | shape 守卫前先 flatten（根除 `[B,1]×[B]`→`[B,B]` 广播放大）；CE labels ∈[0,S) 范围守卫（-1 掩码之外）；`lambda_scale(epoch≥1)` |
| P2 | `frame_selection.py` | bool/非 int 拒绝；budget==1 拒 downsampling 伪装；选中数精确不变量 |
| P2 | `tests/_eva_di_synth.py` | 夹具回填**真实** csv sha 与真实 of3_root（原先全空串→漂移路径根本不可测） |
| P2 | `configs/eva_di/*.yaml` | base/extract 增加显式 `behavior_norm: reuse`；extract 配置注明抽取覆盖全 split（label-free） |

### 10.3 测试面变化
eva_di 套件 127→146（+19）。关键新增：P0-A 差分回归锁、`run_training` CPU 正向学习冒烟（头权重实际更新、metrics.jsonl 行数、run_dir 重跑重置、严格 JSON）、export 往返+三重守卫（mode/ablation/cache provenance）、config 八项类型/一致性守卫与 base 漂移检测、cache 指纹/篡改/of3_root 钉、dataset 截断与方言与 csv-sha 漂移、derange 双射语义三测试、runner 陈旧 manifest/失败落盘/csv 漂移/smoke cap/light 守卫非空洞、metrics 有限性与 shape 纪律。测试侧钉死项同步更新（derange 语义、`EvaDiConfigError`、cache_writer 结构守卫、lambda epoch≥1、provenance warmup 键）。

### 10.4 验证（全部实测）
- `pytest tests -k eva_di -q` → **146 passed**
- `pytest tests -q` → **4 failed, 729 passed, 2 errors**——失败/错误集与冻结基线**逐名一致**（legacy 模块，与本路线无关；710→729P 即新增 19）
- 9/9 `configs/eva_di/*.yaml` 加载通过，`source_sha256 != resolved_sha256`
- E2E 复跑（见 10.5）

### 10.5 E2E 复跑（`/tmp/eva_di_e2e/` 边界不变）
新校验器首先按其设计拦截了 Round-1 的 3-epoch 设置（warmup=5 永不达满强度）→ E2E 配置升为 6 epochs。抽取：**4 条复用**（stale-guard/csv-sha 对账通过）+ **98 条 val 新抽**，124.6 s、391.2 f/s；训练：`epochs_run=6, val_n=100, best_defined=true`——val 决策集完整后 CCC 自 epoch1 起可定义，首次正常选中 **best.pt**（Round-1 因 val_n=2 CCC 恒 NaN 走 last.pt 回退）；epoch 记录含 `warmup_scale=0.2, lambda1_eff=lambda3_eff=0.01`（=λ·w，供 scale² 决策后复核）；导出：`p_mean_val/v_h0_val` 形状 **(100,192)**、train (2,192)，npz 携带 checkpoint sha 且形状断言通过；Round-1 在 export 处的崩溃（state_dict unexpected head keys）已消失。`E2E_OK`。

### 10.6 待用户决策（代码未擅自改语义）
① GRL λ 双重 warmup（λ·w²）保留还是改单段（去掉 `set_lambda` 侧或 CE 侧之一）——影响 REF↔T1↔T3↔FULL↔DERM 单变量可比性；② DERM 双射（现实现）vs 逐样本 shuffle；③ 身份粒度 `[:5]`（session，跨 split 零重叠）vs `[:3]`（person，train∩val 26 人重叠→需 split-wise attacker 设计）；④ recompute==reuse v1 双路并存是否保留；⑤ `behavior_status` 门控。

### 10.7 依赖与解码器说明
**零安装**。PIL 保留为钉死解码器：cv2/torchvision/imageio/av 虽已在 `light` 环境中，但 libjpeg 构建差异会改变解码像素、破坏 oracle 等价门与逐字节 sha 语义；实测吞吐 ~428.8 f/s（解码 1–2 ms/帧），非瓶颈。若未来替换解码器，将作为独立披露包进行像素级等价验证。

### 10.8 更正注记
行为维数为 **206**（196 landmark 轴 + gaze 2 + au 8）；此前主线程部分委派提示中出现的"266"为笔误，代码/contracts 自始为 206，无产品影响。

### 10.9 用户裁决落地（2026-09-05 晚，同包内）
| 裁决 | 落地 |
|---|---|
| **① 仅一次 warmup** | 斜坡 `w_k(e)` 只乘 **CE 权重**侧（`losses.total_loss_terms`，即计划 L61 公式的字面位置）；`set_lambda` 改传**常数** `lam`。恒等强度 = `w_k(e)·lam·CE`（线性，不再是 `lam·w²`）。epoch 记录仍含 `warmup_scale`，`lambda{1,3}_eff` 语义改为常数 λ；config 的 t1==t3 warmup 钉与"warmup≤epochs_max"校验保留。`set_lambda` 钩子保留（设计 §5 接口不变） |
| **② smoke 允许更多 recordings** | E2E 配置 `data.max_recordings: 2→8`（仅截断 train；val 仍全量 100） |
| **③ 身份粒度 = 3** | 新增 `contracts.IDENTITY_PREFIX_LEN = 3`，恒等类 = **person**（`203`）；`subject_of`（`[:5]`）**保留给 BDI 标签路径**——实测 `depression_labels/` 为 150 个 **session 级**文件（`203_1_Depression.csv`），与 `src/datasets/dataset.py:276-280` 旧契约逐字节一致，两粒度自此显式分离。person 跨 split 重叠（train∩val=26）为 AVEC2014 固有设定，即外部身份攻击者的被测现象本身；标签、目标统计、severity 权重仍严格 train-only、session 粒度，无标签泄漏路径。CE 类数由 50（session）变为 41（train person） |
| 仍待决策 | ②' DERM 打乱方式（现=类空间双射）；④' behavior 归一化 reuse/recompute 双路；⑤' `behavior_status` 门控——见最终汇报的通俗版说明 |
| 验证 | `pytest -k eva_di` 146 passed；全仓 4F/729P/2E 与冻结基线一致；E2E（8 train + 100 val、6 epochs、person 粒度、单 warmup）`E2E_OK`（日志 `/tmp/eva_di_e2e/e2e_console_r3.log`） |

### 10.10 用户裁决（第二批，2026-09-05 晚）与实证
| 问题 | 裁决 | 落地 / 证据 |
|---|---|---|
| **DERM 打乱什么？** | 用户澄清：仅打乱**身份标签**则允许 A（整体换人双射）。 | 代码事实（`train.py:246-255`）：`dataclasses.replace(r, subject=...)` **只改 subject 字段**；抑郁标签 `bdi_score/bdi_norm` 与回归损失完全不触碰 → 即"前者"，**A 已生效，零改动** |
| **归一化校验影响效率** | 允许用 OF3 钉死统计（reuse），但校验不得拖慢实际运行。 | `train.py`：全 train split 重算对账**仅在 `behavior_norm: recompute` 审计模式执行**；`reuse`（生产默认）直接使用钉死统计（其 sha 已入 provenance），启动路径零额外 csv 读取。`behavior.py` docstring 与两模式语义对齐；新测试 `test_reuse_mode_skipped_full_recompute_audit` 钉死（reuse→"skipped"，recompute→"audit PASSED"） |
| **OF"状态成功/失败"是什么** | 用户要求解释。 | 实测语义（全量 300 视频 / 493,141 行扫描）：状态列是 OpenFace3 每帧**处理环节的诊断标签**——`quality_status`(检出/置信：ok, low_confidence 2247, failed_detection 14)、`alignment_status`(对齐到 112×112：ok/failed 25/not_run_*)、`landmark_status`、`behavior_status`(AU/视线是否计算：ok/not_run_quality_invalid 2246/not_run_alignment_failed 25)。**蕴含检验：状态非 ok 却声称 `behavior_valid=True` 的行 = 0** → 现有 `behavior_valid`+isfinite 掩码已完全覆盖，行为侧状态门纯冗余；唯一差异是 2232 行（0.46%）`low_confidence` 帧 `image_valid=True`（人脸真实但置信度低，现进入视觉缓存、被排除于行为流）。**裁决建议：不加状态门**（剔除将改变冻结缓存的帧分布且证据不支持）；如需剔除低置信帧属缓存重抽包，暂不启用 |

验证：`pytest -k eva_di` **147 passed**；全仓 **4 failed / 730 passed / 2 errors**（与冻结基线逐名一致，passed 710→730 为新增测试）；E2E 复跑（reuse 快速路径）**108 条缓存全复用、零新抽**，summary 记录 `"consistency recompute skipped on the run path (user decision 2026-09-05)"`，`best_defined=true, val_n=100`，导出 (2,192)/(100,192)，`E2E_OK`（`/tmp/eva_di_e2e/e2e_console_r4.log`）。

### 10.11 `EVA-DI-CODE-v2` 落地（实现缺口补全，本地试运行，2026-09-05）
用户指令"继续补足、落地可运行代码，服务器正式实施"。新增代码产物（全部 CPU/合成测试，本环境不跑真 oracle/真训练）：

| 文件 | 内容 | 验证 |
|---|---|---|
| `src/eva_di/identity_metrics.py` | **外部身份三件套**，直接消费 `export_embeddings` 的 npz（`embeddings[N,192]`+`sample_ids`，subject=`video_id[:3]`）：① fresh **LOVO Ridge** attacker top1/top3（内联 ridge-on-indicators 闭式解，非 sklearn 依赖，逐位可复现；复用冻结 `subject_attacker.compute_attacker_metrics` 行 schema）② identity **pair-AUROC**（内联 Mann-Whitney 中位秩）③ **A1** same-person top1/top3（对角 -1 排除自身，沿用冻结 `identity_retrieval` 语义）。拒绝 test split、拒绝混合 checkpoint | 单测 7 项：聚类嵌入全可解码(top1=1/AUROC>0.999)、噪声嵌入近机会、单人无配对记 unseen 不猜、ridge 与 sklearn `RidgeClassifier` 决策**秩等价**（sklearn 用 ±1 编码故为其 2×−1 仿射）、schema/test/混 ckpt 三拒绝门 |
| `src/eva_di/extract/oracle_equivalence.py` | `raise` 骨架 → 真实运行体：我方 PIL+bf16/none 编码 vs **子进程** trans 参考链（`cwd/PYTHONPATH` 指向 trans，两 `src` 树不冲突，训练/推理期零 trans 导入），逐帧 `cos(cls)>=0.999` 落 `cache_root/_equivalence`（成败皆留证）；`our_encode`/`ref_encode` 可注入桩 | 单测 4 项：恒等→PASSED、扰动→FAILED 仍留证、shape 不匹配拒绝、骨架向后兼容 |
| `src/eva_di/matrix.py` + `configs/eva_di/matrix_v1.yaml` | 消融矩阵**规划器**：spec(数据)→有序计划(arm→config→run_id→train/export/identity 命令)，按现有 `summary.json` 标 done/ready/blocked；stage-2 配对 seeds 缺 derived config 时标 blocked **不伪造**。`--launch` 才顺序训练（停于首败） | 单测：done/seed 失配/config 缺失/dup run_id 四态、stage-2 覆盖与未知 arm 拒绝、dry-run 绝不 spawn（monkeypatch 断言） |
| `src/eva_di/matrix_report.py` | 跨 run 汇总 CCC/MAE + 两外部身份指标成 CSV+MD；**缺失身份证据留空非置零** | 单测：缺 v_h0 证据→该列空、非 run 目录忽略 |
| `scripts/eva_di/eva_di_run.sh` | 服务器一键 `{extract\|oracle\|plan\|launch\|identity\|report}`（本环境不执行） | 语法/`--help` |

**本地试运行**（`/tmp` E2E 边界内，真实导出嵌入，CPU）：
- identity CLI 于 `e2e-20260905-1`（未防御 smoke）：exit-P LOVO top1=0.66/top3=0.84、pair-AUROC=0.943、A1 top1=0.74；exit-V top1=0.47、pair-AUROC=0.945——与"身份强线性可解码"的既有负证据方向一致，门控工具确测到身份泄漏；
- matrix 规划对 7 个真实 config 全 ready；matrix_report 对真实 run 出全指标表。

验证：`pytest -k eva_di` **162 passed**；全仓 **4 failed / 745 passed / 2 errors**（失败/错误名与冻结基线逐名一致，passed 730→745 为 15 项新测试）。回滚：新增文件未跟踪，`rm` 即回；`oracle_equivalence.py` 保留原骨架函数签名向后兼容。
