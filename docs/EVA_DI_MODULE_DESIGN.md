# EVA_DI_MODULE_DESIGN.md

> 文档职责：`PLAN-EVA-DI-VALIDATE-v1` 的设计附件（docs-only）。在`EVA_DI_VALIDATION_PLAN.md`（事实基线与实施契约）之上，钉死`eva_di`模块的目录与所有权、模块公共接口、配置schema、路径与环境变量约定、磁盘布局、异常语义、测试约定与复用边界。本文档不创建任何可执行产物；`EVA-DI-CODE-v1`实现必须与本设计逐条一致，任何偏离须先修订本文档。

## 1. 总则

1. **单一真源**：所有schema常量、SHA、维度、路径默认值只允许定义在`src/eva_di/contracts.py`；其余模块只import不复制。
2. **运行时仓库零写入**：任何CLI/库运行期只允许写三类根：`cache_root`（数据盘）、`log_root`、`output_root`；仓库目录在运行期视为只读。
3. **fail-fast**：一切严格读取以显式异常终止并给出字段级差异，禁止静默降级、禁止静默填值、禁止`strict=False`类语义。
4. **环境纪律**：所有CLI首行自检`sys.executable`位于`~/miniconda3/envs/light/`，否则拒绝运行并打印期望值；`HF_HUB_OFFLINE=1`。
5. **test锁定**：v1所有入口的split参数仅接受`{train,val}`；出现`test`直接拒绝。locked benchmark阶段属未来独立包，届时另行设计显式flag。

## 2. 目录与所有权

```text
src/eva_di/
  __init__.py            # 仅导出 __version__ 与 contracts 常量
  contracts.py           # 常量/异常/dataclass契约（单一真源，无副作用）
  paths.py               # 路径注册表 + EVA_DI_* 环境变量覆盖 + sha256_file
  of3_registry.py        # 300视频 features.csv 严格注册表（模式校验/逐视频SHA/一致性断言）
  frame_selection.py     # uniform512_v1 纯函数（无IO、确定性）
  behavior.py            # 206d行为解析 + train-only统计装载与重算一致性校验
  subject_table.py       # train-only subject类别表（video_id[:5]）
  extract/
    __init__.py
    encoder.py           # EVA02冻结编码器（权重钉验、transform、分块no_grad+bf16）
    cache_writer.py      # eva_di_frame_feature_cache_v1 原子写入 + manifest + 续跑
    runner.py            # 抽取驱动（10视频smoke先行、逐视频记录、runner日志）
    oracle_equivalence.py# 与trans冻结路径逐帧对照（属EVA-DI-EXTRACT-v1，CODE包只留骨架）
  cache_reader.py        # 严格读取器（manifest指纹字段级比对、全量驻留）
  dataset.py             # RecordingIndex：split×标签×缓存×行为 四源装配与交叉断言
  batching.py            # 主进程向量化组批（padding、前缀mask断言、空有效拒绝）
  model.py               # DualStreamDI + IdentityHeads（T1/T2/T3）
  losses.py              # severity加权回归 + 帧/视频CE + warmup调度
  metrics.py             # CCC/MAE/RMSE、severity分组[13,19,28]（nan→"unknown"）
  export_embeddings.py   # 出口P/V嵌入导出（供 src/diagnostics/subject_attacker.py 消费）
  train.py               # 训练CLI（fit→val only、run_provenance.json）
  config.py              # YAML严格装载+校验（未知键即错）
configs/eva_di/          # 见§6（EVA-DI-CODE-v1才创建）
tests/test_eva_di_*.py   # 见§9
```

不触碰：`src/datasets/dataset.py`、`src/models/mtl_lite.py`、`configs/avec2014_base.yaml`及一切既有模块的默认行为。

## 3. `contracts.py` 常量清单

```python
PLAN_ID            = "PLAN-EVA-DI-VALIDATE-v1"
SCHEMA_FRAME_CACHE = "eva_di_frame_feature_cache_v1"
SCHEMA_CONFIG      = "eva_di_config_v1"
SCHEMA_RUNNER_LOG  = "eva_di_runner_log_v1"
SCHEMA_PROVENANCE  = "eva_di_run_provenance_v1"
SELECTION_POLICY   = "uniform512_v1"; FRAME_BUDGET = 512
EVA_BACKBONE       = "eva02_small_patch14_224.mim_in22k"
EVA_WEIGHT_SHA256  = "d3d632352efbd0a0a8269dce114ab8a214833f85ce6f738860221f11ab0f0c2f"
EVA_WEIGHT_SIZE_BYTES = 86_499_188
EVA_WEIGHT_REVISION  = "79c7d4274f6dbf202549d8f976ae24eeaf97e5ad"
OF3_ROOT_DEFAULT   = "/home/zhen/code/trans/outputs/openface3/20260727-avec2014-formal-features-masked-v2"
SCHEMA_ALIGNMENT   = "retinaface_5pt_arcface112_masked_v1"
SCHEMA_BEHAVIOR    = "openface3_wflw98xy_gaze2_anonymous_mtl8_v1"
FEATURES_CSV_COLUMNS = (...42列逐字清单，顺序固定...)
BEHAVIOR_AXIS_ORDER  = ("landmarks_98_xy_aligned", "gaze", "au")  # 196+2+8=206
BEHAVIOR_DIM = 206; EVA_FEATURE_DIM = 384; PROJ_DIM = 192
SEVERITY_BOUNDS = (13, 19, 28)
class EvaDiError(RuntimeError); class EvaDiSchemaError(EvaDiError)
class EvaDiPathError(EvaDiError);  class EvaDiFingerprintError(EvaDiError)
```

## 4. 路径注册表与环境变量

| 逻辑名 | 默认值 | 环境变量覆盖 | 运行期权限 |
|---|---|---|---|
| `of3_root` | `OF3_ROOT_DEFAULT` | `EVA_DI_OF3_ROOT` | 只读 |
| `avec_root` | `/home/zhen/dataset/depression/avec/2014` | `EVA_DI_AVEC_ROOT` | 只读（除`cache_root`子目录） |
| `split_file` | `<avec_root>/dataset_split.json` | `EVA_DI_SPLIT_FILE` | 只读，装载时记SHA进manifest |
| `label_dir` | `<avec_root>/depression_labels` | `EVA_DI_LABEL_DIR` | 只读 |
| `cache_root` | `<avec_root>/eva02_features_v1` | `EVA_DI_CACHE_ROOT` | EXTRACT包唯一可写数据目标 |
| `weight_path` | `…/artifacts/eva02_small_patch14_224.mim_in22k/model.safetensors`（解包后） | `EVA_DI_WEIGHT_PATH` | 只读+SHA校验 |
| `behavior_norm_root` | `<of3_root>/behavior_normalization_train_v1` | `EVA_DI_BEHAVIOR_NORM_ROOT` | 只读 |
| `log_root` | STC仓库`logs/eva_di` | `EVA_DI_LOG_ROOT` | 可写 |
| `output_root` | STC仓库`outputs/eva_di` | `EVA_DI_OUTPUT_ROOT` | 可写（不入git，`.gitignore`核对） |

`paths.py`解析后返回冻结`PathSet` dataclass；`resolve()`时对每个只读根做存在性+（有钉死值时）SHA断言。

## 5. 公共接口签名（模块间唯一耦合面）

```python
# frame_selection.py —— 确定性纯函数
def uniform_select(n_frames: int, budget: int = FRAME_BUDGET) -> np.ndarray:  # int64, 严格递增, 含首尾
# 算法钉死: n<=budget -> np.arange(n)
# 否则 np.unique(np.rint(np.linspace(0, n-1, budget)).astype(np.int64))
# 断言: len>=budget//2; 幂等: 对已选帧再抽取仍含首尾

# of3_registry.py
def load_video_record(of3_root: Path, sample_id: str, verify_sha: bool = True) -> VideoRecord
@dataclass(frozen=True) class VideoRecord:
    sample_id: str; split: str; n_rows: int
    frame_index_zero: np.ndarray      # 全行
    image_valid: np.ndarray           # bool, 全行
    behavior: np.ndarray              # [n_rows,206] float32, 非有限值位置置0.0且behavior_valid=False
    behavior_valid: np.ndarray; pair_valid: np.ndarray
    features_sha256: str
# 校验: 42列逐字匹配; schema列值==SCHEMA_BEHAVIOR/SCHEMA_ALIGNMENT; frame_index_zero严格递增;
#       split∈{train,dev,test,val}(仅断言, 不作split来源); 违规 -> EvaDiSchemaError(含列名/行号)

# behavior.py
def load_train_stats(norm_root: Path) -> BehaviorStats   # 断言 behavior_schema==SCHEMA_BEHAVIOR, mean/std长度==206
def assert_stats_consistency(stats, registry_train_only, tol=1e-4) -> None  # train行重算, 超差 -> EvaDiFingerprintError

# subject_table.py
def build_subject_table(split_videos: dict[str, list[str]], subject_of=lambda v: v[:5]) -> SubjectTable
# SubjectTable: classes仅来自train split; __call__(video_id)->int64|−1(未映射, 样本级剔除用)

# extract/encoder.py
class Eva02FrozenEncoder:
    def __init__(self, weight_path: Path, device: str = "cuda", chunk: int = 128)
    # 装载前置断言: 文件SHA==EVA_WEIGHT_SHA256且size==EVA_WEIGHT_SIZE_BYTES, 否则EvaDiFingerprintError
    def encode(self, frames_uint8: Tensor[N,3,112,112]) -> tuple[np.ndarray, np.ndarray]
    # 内部: uint8/255→float32→bicubic 224→timm cfg mean/std(运行时读取)→no_grad+bf16分块
    # 返回 cls[N,384] f32, gap[N,384] f32(gap=patch tokens 1..256均值); 输出token结构断言[257,384]
    def metadata(self) -> dict  # sha/revision/timm/torch版本/mean/std/输入契约, 写入manifest

# extract/cache_writer.py
def write_video_cache(cache_root: Path, sample_id: str, sel: SelectionArrays, resume: bool = True) -> ManifestEntry
# SelectionArrays: cls,gap f32[N,384]; frame_index_zero int64[N]; frame_valid bool[N]; meta dict
# 纪律: .{name}.tmp原子写+rename; 逐文件SHA256; resume=既有文件SHA与manifest一致则跳过

# cache_reader.py
class FrozenFrameCache:
    def __init__(self, cache_root: Path, expected: CacheFingerprint, verify_hashes: bool = True)
    # CacheFingerprint字段级比对(逐项列差异后抛错): schema/selection_policy/frame_budget/backbone/
    #   weight_sha256/weight_revision/timm_version/feature_dim/feature_dtype + counts对账
    def load_resident(self, sample_ids: Sequence[str]) -> ResidentArrays   # 全量驻留; 缺样本 -> EvaDiPathError

# dataset.py
def build_recording_index(cfg: EvaDiConfig) -> RecordingIndex
# 四源交叉断言: split_file视频集 == cache manifest视频集 == of3 registry视频集(逐一, 差异列表化)
# 标签: <avec_root>/depression_labels/<subject_session>_Depression.csv, 契约同 src/datasets/dataset.py:276-307
# Recording: sample_id, split, subject(int64|-1), bdi(float32), cls/gap/behavior/frame_mask(=pair_valid∧选中)

# batching.py
@dataclass class DIBatch:
    frames_p_in: Tensor[B,T,384|768]; behavior: Tensor[B,T,206]; frame_mask: Tensor[B,T] bool
    bdi: Tensor[B]; subject: Tensor[B] long; sample_ids: list[str]
def collate(recordings, device) -> DIBatch
# 尾padding; 断言frame_mask为前缀True后缀False; 任一视频0有效帧 -> EvaDiSchemaError(拒整批并点名)

# model.py
class DualStreamDI(nn.Module):
    def __init__(self, cfg: ModelCfg)   # behavior_mlp(206->LN->64) / projector(384|768+64 ->192) / GRU(192,1层)
    def forward(self, batch: DIBatch) -> DIForward    # p[B,T,192], h0[B,192], bdi_pred[B]
    def identity_logits(self, out: DIForward, batch) -> tuple[Tensor[B*Tv, S]|None, Tensor[B, S]|None]
    # 仅 self.training and cfg.enable 时构造/前向; GRL复用 src/models/gradient_reversal.py:31 GradientReversalLayer
    # head懒建后立即 .to(device); λ经 set_lambda() 由runner按warmup逐epoch注入

# losses.py
def severity_weighted_l2(pred, target, table_f32, group_of) -> Tensor     # 表固定float32构造
def ce_masked(logits, subject, frame_mask=None, label_smoothing=0.1) -> Tensor
    # subject==-1样本级剔除(不整批置零); frame logits只在mask=True帧计CE
def lambda_scale(epoch: int, warmup_epochs: int) -> float                  # min(1, epoch/warmup)

# metrics.py
def ccc/mae/rmse(y, yhat) -> float
def severity_group(score) -> str   # <13 mild | <19 moderate | <28 severe | >=28 very_severe | nan->"unknown"

# export_embeddings.py  (CLI)
python -m eva_di.export_embeddings --run-dir <dir> --split {train,val}
# 写 <run-dir>/embeddings/p_mean_<split>.npz (每录音p有效帧均值 [N,192]) 与 v_h0_<split>.npz
# 消费方: src/diagnostics/subject_attacker.py (fresh LOVO Ridge), pair-AUROC, A1检索 —— 模型内无身份指标
```

## 6. 配置schema（`configs/eva_di/`，`eva_di_config_v1`）

严格装载器`config.py`：未知键→`EvaDiSchemaError`；下表即全集；默认值仅存在于dataclass，YAML必须显式给出`paths/run/extraction`三段。

```yaml
schema_version: eva_di_config_v1
run:   {mode: train, seed: 42, run_id: null}          # run_id空则 <yyyymmdd>-<config名>-seed<seed>
paths: {of3_root, avec_root, cache_root, weight_path, log_root, output_root}   # 允许${EVA_DI_*}
extraction:  # 仅EXTRACT入口读取; train入口断言与cache manifest一致后忽略
  backbone: eva02_small_patch14_224.mim_in22k
  weight_sha256: d3d632...0c2f        # 与contracts一致性再断言(双保险)
  chunk: 128; autocast: bf16; input_size: 224; selection_policy: uniform512_v1; frame_budget: 512
data:
  splits: [train, val]                # test出现即拒
  behavior_norm: reuse                # reuse|recompute; 两值都要求一致性断言通过
model:
  use_gap: false; proj_dim: 192; gru_hidden: 192; gru_layers: 1
  behavior_mlp_dim: 64; behavior_dropout: 0.2; proj_dropout: 0.1
identity:
  t1: {enable: false, weight: 0.05, lam: 0.05, warmup_epochs: 5, label_smoothing: 0.1}
  t3: {enable: false, weight: 0.05, lam: 0.05, warmup_epochs: 5, label_smoothing: 0.1}
ablation: {input_mode: dual}          # dual|eva_only|of3_only; 关闭流=投影支路置零输入且行为统计仍train-only
training:
  precision: bf16-mixed; optimizer: {name: adamw, lr: 3e-4, weight_decay: 1e-2}
  batch_recordings: 8; epochs_max: 60
  early_stop: {metric: val_ccc, patience: 10}    # 只看val
  grad_clip_norm: 1.0
```

矩阵配置（DI-REF/T3/T1/FULL/DERM、EVA-only、OF3-only）由base做键级override，全部落`configs/eva_di/`，不改任何既有STC配置文件。

## 7. 磁盘布局（运行产物合同）

```text
<cache_root>/
  cache_manifest.json                    # 见下; 写序: 先全部样本条目 -> 最后原子写manifest
  <sample_id>/{cls.npy, gap.npy, frame_index_zero.npy, frame_valid.npy, meta.json}
  _equivalence/                          # oracle对照证据(EXTRACT包), 不入任何指标集
cache_manifest.json 必需字段:
  schema_version, selection_policy, frame_budget, backbone, weight_sha256, weight_revision,
  timm_version, torch_version, cuda_version, input_size, feature_dim, feature_dtype,
  of3_root, features_csv_sha256_by_sample{...}, sample_count, n_rows_total,
  n_selected_total, n_valid_total, n_exact_zero_masked, samples{<id>:{cls_sha256, gap_sha256,
  index_sha256, valid_sha256, n_frames, meta_sha256}}
<output_root>/<run_id>/
  run_provenance.json    # SCHEMA_PROVENANCE: git commit/branch/dirty、argv、seed、库版本、
                         # device、cache_manifest.json sha、split_file sha、resolved config sha
  config_resolved.yaml; metrics.jsonl(epoch级train/val); val_metrics.json; checkpoints/; embeddings/
<log_root>/runner_<UTC>.json             # SCHEMA_RUNNER_LOG: accepted/jobs_sha256/逐视频status/sha/吞吐(镜像trans runner格式)
```

## 8. 异常与降级矩阵（钉死）

| 场景 | 行为 |
|---|---|
| features.csv列缺失/顺序错/schema值不符 | `EvaDiSchemaError`，含列名与首个违规行号 |
| aligned jpg缺失或SHA不符（EXTRACT期） | 该帧`frame_valid=False`并计数；比例>1%则整视频failed并停止runner |
| exact-zero黑帧 | `frame_valid=False`（先于编码），manifest计数 |
| 行为值非有限 | `behavior_valid=False`，数组位置0.0且永不进统计/CE/reg输入 |
| cache manifest指纹不符 | `EvaDiFingerprintError`，字段级差异列表 |
| train重算统计与钉死json偏差>tol | `EvaDiFingerprintError`，不静默切换 |
| split出现test / 决策CLI触test | 参数解析层拒绝 |
| 权重SHA/size不符 | 拒绝装载，无回退路径 |
| 视频0有效帧 | collate层`EvaDiSchemaError`（点名sample_id） |

## 9. 测试约定

- 命名`tests/test_eva_di_<area>.py`；`area ∈ {contracts, frame_selection, of3_registry, behavior, subject_table, cache_writer, cache_reader, dataset, batching, model, losses, metrics, config, provenance}`。
- 单元测试全部用`tmp_path`合成fixture（小型伪features.csv/伪缓存），**不读真实数据盘**；真实数据端到端用例挂`@pytest.mark.eva_di_data`（默认skip，EXTRACT包后手动启用）。
- 必测断言：`uniform_select`黄金值与幂等；GRL梯度方向（输入梯度符号取反、head参数按CE正常更新）；前缀mask不变量与0有效帧拒绝；NaN→mask且不进统计；subject==-1样本级剔除；默认位（t1/t3 disable）与数学参照bit-close；severity nan→"unknown"；config未知键拒绝；writer原子性/续跑（SIGKILL模拟可后置）。
- 验收命令（CODE包）：`~/miniconda3/envs/light/bin/python -m pytest tests -k eva_di -q` + `python -c "import eva_di"` + config装载检查；全部CPU。

## 10. 复用与边界

- **复用（import）**：`src/models/gradient_reversal.py:GradientReversalLayer`（语义不变）；`src/diagnostics/subject_attacker.py`及其口径作为外部身份判定唯一入口（eva_di只导出npz，不调用其进训练环）。
- **不import**：trans仓库任何代码（其AGENTS §2禁区+运行时耦合）；`models/image.py`的transform逻辑以**注明来源的等价重写**入`encoder.py`，正确性由oracle对照证明（cos≥0.999，3视频×8帧，`_equivalence/`留证）。
- **只读资产冻结清单**：OF3 300视频目录、`behavior_normalization_train_v1`、`dataset_split.json`、`depression_labels`、EVA02权重。发现内容变化（SHA漂移）→ 停止并报告，不更新常量。

## 11. 验证（本docs附件）

- 仅新增本文件 + `EVA_DI_VALIDATION_PLAN.md`头部一行设计附件指针；未创建`src/configs/tests/scripts`任何文件，未运行任何数据/GPU/训练程序。
- 待办（不在本附件）：`features.csv` 42列逐字清单在CODE包`contracts.py`落地时逐字抄录并加单测锁列；`behavior_normalization_train_v1`各json键名在CODE包只读核对后写入`behavior.py`装载器断言。
