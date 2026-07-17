# 实验脚本使用手册

本手册汇总当前项目所有可直接运行的训练、诊断、审计和汇总脚本。所有路径默认采用新的实验输出布局：

```text
<LOG_DIR>/<EXPERIMENT_GROUP>/<EXPERIMENT_NAME>/version_N/
```

- `LOG_DIR` 在 `configs/local_paths.yaml` 中设置（机器相关，不提交 git）。
- `EXPERIMENT_GROUP` / `EXPERIMENT_NAME` 来自 base/override 配置。
- `version_N` 由 `src/config.py` 自动按已有目录递增（`version_0`、`version_1`、...）。

---

## 1. 配置栈

实验配置按以下顺序合并：

1. `configs/avec2014_base.yaml`（MTL-Lite 基座最小配置）
2. `configs/local_paths.yaml`（机器相关路径，**不提交到 git**，从 `configs/local_paths.example.yaml` 复制）
3. 一个或多个 override YAML（如 `configs/mtl_lite_baseline.yaml`）

显卡选择通过 `DEVICES` 列表显式指定，例如 `DEVICES: [0]` 表示使用 0 号卡，`DEVICES: [0, 1]` 表示使用 0、1 号卡。默认 base 配置为 `[0]`；如需换卡，在 override 或 `configs/local_paths.yaml` 中覆盖即可。

创建本地路径配置：

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
# 编辑 configs/local_paths.yaml 填入真实路径
```

`LOG_DIR` 是新实验的主要输出根目录，实际路径为 `<LOG_DIR>/<EXPERIMENT_GROUP>/<EXPERIMENT_NAME>/version_N/`。

---

## 2. 训练入口

### 2.1 准备本地 backbone 权重（推荐）

```bash
python scripts/prepare_backbone_weights.py \
  --model-name deit_tiny_patch16_224 \
  --timm-model-name deit_tiny_patch16_224.fb_in1k \
  --output weights/deit_tiny_patch16_224/model.pth \
  --verify
```

然后在 override 中指定：

```yaml
EXTRACT_FEATURE:
  MODEL_NAME: "deit_tiny_patch16_224"
  TIMM_PRETRAINED: False
  MODEL_WEIGHT_PATH: "weights/deit_tiny_patch16_224/model.pth"
```

### 2.2 MTL-Lite baseline

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_baseline.yaml
```

输出：

```text
<LOG_DIR>/default/mtl_lite/version_0/
  metrics.csv
  hparams.yaml
  resolved_config.yaml
  checkpoints/
```

### 2.3 Regression-only baseline

```bash
python scripts/train_mtl_lite.py --override configs/regression_only_baseline.yaml
```

输出：

```text
<LOG_DIR>/default/regression_only/version_0/
```

### 2.4 MTL-Lite debug smoke

```bash
python scripts/train_mtl_lite.py --override configs/mtl_lite_debug_smoke.yaml
```

输出：

```text
<LOG_DIR>/default/mtl_lite_debug_smoke/version_0/
```

### 2.5 Explicit L1/L2 regularization audit

该审计在 Stage B regression-only 协议上固定比较四组完整 40 epoch 运行：

```text
reference: L1=0,    L2=0
l1:        L1=0.01, L2=0
l2:        L1=0,    L2=0.1
elastic:   L1=0.01, L2=0.1
```

AdamW `PROCESS_TEMPORAL.WEIGHT_DECAY` 不变；显式参数惩罚只进入 train loss，
验证和测试指标不包含惩罚。运行矩阵并汇总 overfit 曲线：

```bash
bash scripts/regularization_audit/run_matrix.sh
```

仅运行短 smoke：

```bash
DEBUG=1 SKIP_DIAG=1 bash scripts/regularization_audit/run_matrix.sh
```

已有 run 只做汇总：

```bash
SKIP_TRAIN=1 SKIP_DIAG=1 bash scripts/regularization_audit/run_matrix.sh
```

### 2.6 Continuous severity-density weighting

该审计用 train-only BDI 标签直方图估计连续密度，经 Gaussian 平滑后对回归 MSE
赋予连续权重。首轮只比较 `POWER=0.25` 和 `POWER=0.5`，固定
`SMOOTHING_SIGMA=2.0`、`DENSITY_EPSILON=0.001`、权重范围 `[0.5, 4.0]`。

```bash
bash scripts/continuous_severity_weighting/run_matrix.sh
```

短 smoke：

```bash
DEBUG=1 SKIP_DIAG=1 bash scripts/continuous_severity_weighting/run_matrix.sh
```

权重只作用于 BDI regression MSE；CCC、ordinal、validation/test 标签和测试指标口径不变。
结果应与 `logs/stage_b/aggregate/` 中已有 E2 四档权重结果比较。

### 2.7 Stage C reference

Stage C 必须按 `common -> experiment -> seed -> optional debug` 顺序叠加 override。`C-REF` reference：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_ref_e2.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```

短 smoke 在最后叠加：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_ref_e2.yaml \
  --override configs/stage_c/seeds/seed_42.yaml \
  --override configs/stage_c/debug_smoke.yaml
```

`C-BN` debug smoke：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_bn_bottleneck.yaml \
  --override configs/stage_c/seeds/seed_42.yaml \
  --override configs/stage_c/debug_smoke.yaml
```

Train-only calibration 不叠加 debug override；runner 固定执行 100 train steps，并关闭 validation/test：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/calibration_train_only.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```

`C-REC/C-FULL` 已使用 train-only calibration 冻结权重。seed-42 utility gate 已失败，当前不进入 C3；完整结论和停止条件见 `docs/STAGE_C_RUNBOOK.md`。

配置和 P0 策略验证：

```bash
python -m pytest tests/test_mtl_lite_training_policy.py tests/test_stage_c_config_contract.py
python -m pytest tests/test_task_nuisance.py tests/test_mtl_lite_forward.py tests/test_mtl_lite_loss_backward.py
```

### 2.8 Behavior-only baseline

需要额外提供 `DATASET.OPENFACE_ROOT`：

```bash
python scripts/train_behavior_baseline.py \
  --override configs/behavior_baseline.yaml \
  --override configs/local_openface_paths.yaml
```

示例 `configs/local_openface_paths.yaml`：

```yaml
DATASET:
  OPENFACE_ROOT: "/path/to/openface_csv_root"
```

输出：

```text
<LOG_DIR>/default/behavior_baseline/version_0/
  metrics.csv
  checkpoints/
  diagnostics/behavior/val_predictions.csv
  diagnostics/behavior/test_predictions.csv
```

### 2.9 Validation/Test role-swap split-sensitivity audit

该实验保持 train、seed、backbone、optimizer、损失和 EarlyStopping 不变，
只交换逻辑 `val` 与 `test` 的物理来源。它是 exploratory split-sensitivity
审计，不是新的无偏测试集结果；原 test 在交换条件下参与 checkpoint 选择。
完整协议见 `configs/split_sensitivity/README.md`。

```bash
python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml \
  --override configs/split_sensitivity/reference.yaml

python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml \
  --override configs/split_sensitivity/val_test_swapped.yaml
```

短 smoke 在最后叠加 `configs/split_sensitivity/debug_smoke.yaml`。交换条件
的映射为：logical `train` = original `train`，logical `val` = original
`test`，logical `test` = original `val`。

---

## 3. 离线诊断入口（`diagnose_mtl_lite.py`）

### 3.1 默认只诊断 test split

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/mtl_lite/version_0 \
  --ckpt best
```

输出：

```text
<LOG_DIR>/default/mtl_lite/version_0/diagnostics/
  regression/test_predictions.csv
  embeddings/test_features.npz
  training/training_curves.png
  correlation/...
  reports/diagnostic_report.md
```

### 3.2 同时诊断 val 和 test

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/mtl_lite/version_0 \
  --ckpt best \
  --split val test
```

输出：

```text
<LOG_DIR>/default/mtl_lite/version_0/diagnostics/
  val/regression/val_predictions.csv
  val/embeddings/val_features.npz
  val/reports/diagnostic_report.md
  test/regression/test_predictions.csv
  test/embeddings/test_features.npz
  test/reports/diagnostic_report.md
  reports/diagnostic_summary.md
```

### 3.3 只跑部分诊断（按需开启）

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/mtl_lite/version_0 \
  --ckpt best \
  --split test \
  --enable-regression \
  --enable-embeddings \
  --enable-training-curves
```

### 3.4 Stage C 只读失败机制分析

先用 `diagnose_mtl_lite.py --split train val --enable-embeddings` 为四组导出 NPZ/CSV，再运行：

```bash
python scripts/audit_representation_leakage.py \
  --run C-REF <TRAIN_NPZ> <VAL_NPZ> \
  --run C-BN <TRAIN_NPZ> <VAL_NPZ> \
  --run C-REC <TRAIN_NPZ> <VAL_NPZ> \
  --run C-FULL <TRAIN_NPZ> <VAL_NPZ> \
  --output-dir logs/stage_c/failure_analysis/representation_leakage

python scripts/audit_group_robustness.py \
  --run C-REF <TRAIN_PRED> <VAL_PRED> \
  --run C-BN <TRAIN_PRED> <VAL_PRED> \
  --run C-REC <TRAIN_PRED> <VAL_PRED> \
  --run C-FULL <TRAIN_PRED> <VAL_PRED> \
  --output-dir logs/stage_c/failure_analysis/group_robustness
```

两个脚本不加载或修改 checkpoint，只读取冻结的 train/val 导出。连续风险轴必须同时提供 train/val weak-label CSV；阈值只从 train 中位数估计。

### 3.5 Full-40 Bottleneck Capacity Audit

容量诊断使用独立实验组，关闭 EarlyStopping 以观察完整 40 epoch 曲线，但仍不运行 test：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_c/common.yaml \
  --override configs/stage_c/c_bn_bottleneck.yaml \
  --override configs/stage_c/capacity_audit/common_full40.yaml \
  --override configs/stage_c/capacity_audit/dep_128.yaml \
  --override configs/stage_c/seeds/seed_42.yaml
```

将 `dep_128.yaml` 依次替换为 `dep_96.yaml`、`dep_160.yaml`、`dep_192.yaml`。四个点必须全部预先固定，不根据结果追加维度。

### 3.6 Regression-only Identity Gradient Audit

该配对实验固定纯 regression-only 基座，比较 task-only reference 与
`L_BDI + GRL(L_identity)`。两组完整运行 40 epoch 用于过拟合审计，但不打开
test；candidate 同时记录 BDI/反向 identity 梯度的范数、比值、余弦、冲突率
和抵消率。

```bash
bash scripts/identity_gradient_audit/run_matrix.sh
```

首次运行先做 candidate debug smoke：

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/identity_gradient_audit/common_full40.yaml \
  --override configs/identity_gradient_audit/bdi_identity_adversarial.yaml \
  --override configs/identity_gradient_audit/debug_smoke.yaml
```

只有 debug `metrics.csv` 出现有限的 `train_grad_*_step/epoch` 字段后，才运行
full-40 配对矩阵。

只复用已完成 run 并重新汇总：

```bash
SKIP_TRAIN=1 bash scripts/identity_gradient_audit/run_matrix.sh
```

只生成 gradient conflict 与 training overfit，不重新导出 train/val 表征：

```bash
SKIP_TRAIN=1 SKIP_DIAG=1 \
  bash scripts/identity_gradient_audit/run_matrix.sh
```

输出位于 `<LOG_DIR>/identity_gradient_audit/analysis/`，包括
`gradient_conflict`、`training_overfit`、`representation_leakage` 和
`group_robustness`。完整协议与字段解释见
`configs/identity_gradient_audit/README.md`。读取结果前不得追加 lambda 或维度点。

---

## 4. 审计入口

所有审计脚本都是**只读**的，不改变训练/验证/测试结果。输入为 prediction CSV 或 OpenFace/aligned frame 元数据。

### 4.1 Split / subject integrity audit

```bash
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/split_integrity
```

### 4.2 Temporal sampling audit

> 2026-07-17命名说明：下方`AU-T0*`是既有脚本/输出名。当前项目不读取AU intensity/presence数值、AU序列或AU特征；这些命令用于frame、landmark coordinate、failure、presence、exposure和区域定位审计。当前任务名使用`LM-T0 / FACE-S*`，但局部RGB crop仍按AU/FACS语义分区。

```bash
python scripts/audit_temporal_sampling.py \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/temporal_sampling \
  --sample-step 10 \
  --max-seq-len 2000 \
  --sampling-strategy stride_head
```

`--sample-step` / `--max-seq-len` / `--sampling-strategy` 必须与对应实验的 `resolved_config.yaml` 一致。

### 4.2.1 LM-T0a frame-contract inventory（历史脚本名AU-T0a）

该命令只核对 aligned JPG 与 OpenFace CSV 帧，不生成裁剪图或修改训练数据：

```bash
python scripts/audit_au_region_tracking.py \
  --image-root /path/to/AVEC2014/face_images \
  --openface-root /path/to/openface_csv_root \
  --output-dir logs/au_region_tracking_audit/t0a_frame_contract \
  --sample-step 10 \
  --max-seq-len 2000 \
  --sampling-strategy stride_head \
  --join-threshold 0.995
```

若 JPG 文件名中的帧号不是最后一个数字组，使用例如 `--frame-id-regex 'frame_(\d+)'` 显式指定。参数必须先在 train 数据约定上冻结，不能根据 validation/test 结果反复修改。

### 4.2.2 Aligned JPG server/local integrity gate

在本地重新运行 OpenFace 前，服务器和本地必须分别对训练实际可见的全部 JPG 生成清单。第一轮只运行文件 SHA-256、完整图片解码和 `112x112` 尺寸检查；若得到 `EXACT_PASS`，逐字节一致已经成立，不需要额外计算 decoded-RGB pixel SHA-256。只有出现文件哈希不同时，才在两端附加 `--pixel-hash` 重跑以区分 JPEG 重编码与真实像素变化。

服务器运行：

```bash
python scripts/audit_image_integrity.py inventory \
  --image-root /usr/local/conda/zhen/dataset/AVEC2014/face_images \
  --output-dir logs/au_region_tracking_audit/image_integrity/server \
  --label server \
  --workers 16
```

将 `logs/au_region_tracking_audit/image_integrity/server/` 复制到 Windows 磁盘后，本地 inventory 推荐直接使用 Windows Python + PowerShell。这样 JPG 读取和 Pillow 解码都发生在 NTFS 本地路径，不经过 WSL 文件系统桥接。

先设置实际路径。当前 `$Repo` 是位于 NTFS 的 Windows 代码副本，不包含 `.git`；权威版本仍由 WSL checkout 提供：

```powershell
$Repo = "D:\Project\stc"
# 也可直接读取 WSL 中的脚本，但大量 JPG 仍必须使用 Windows 本地路径：
# $Repo = "\\wsl.localhost\Ubuntu\home\zhen\code\stc"

$ImageRoot = "D:\Project\dataset\AVEC2014\face_images"
$AuditRoot = "D:\Project\dataset\AVEC2014\audits\image_integrity"
$Python = "python"  # 也可替换为 Windows conda 环境中的 python.exe 绝对路径

& $Python -c "from PIL import Image; print(Image.__version__)"

& $Python "$Repo\scripts\audit_image_integrity.py" inventory `
  --image-root "$ImageRoot" `
  --output-dir "$AuditRoot\local" `
  --label windows_local `
  --workers 16
```

如果使用 Python launcher，可将 `& $Python` 替换为 `py -3`。Windows 环境只需要 Python 和 Pillow；若 Pillow 尚未安装，应先在同一 Python 环境中安装并重新运行上面的 import 检查。

把服务器清单放到 `$AuditRoot\server` 后，在同一 PowerShell 中比较两份排序 manifest：

```powershell
& $Python "$Repo\scripts\audit_image_integrity.py" compare `
  --reference-manifest "$AuditRoot\server\tables\image_manifest.csv" `
  --candidate-manifest "$AuditRoot\local\tables\image_manifest.csv" `
  --output-dir "$AuditRoot\comparison"
```

最终门槛为 `EXACT_PASS`：所有训练可见 JPG 的相对路径、文件数量、完整解码、`112x112` 尺寸和文件 SHA-256 均一致。`PIXEL_EQUIVALENT` 表示 decoded RGB 相同但 JPEG 文件字节不同，只能作为可解释的重编码情况，不能声明原始文件逐字节一致。`FAIL` 时先依据 `image_comparison_issues.csv` 修复缺失、额外、损坏或像素不同文件。调试可附加 `--max-videos 2`，正式清单必须删除该参数。

若首次比较只出现 `BINARY_MISMATCH_PIXEL_UNKNOWN`，服务器和 Windows 本地分别在新的输出目录中附加 `--pixel-hash` 重跑 inventory，再比较新 manifest。不要默认对约 49 万张图片计算像素哈希，以减少 RGB 转换和约数十 GB decoded-pixel hashing 开销。

### 4.2.3 OpenFace 2.2.0 aligned-space landmark extraction

当前冻结工具根目录为 `D:\Tools\Openface_2.2.0_win_x64`，实际发布包位于其下的 `OpenFace_2.2.0_win_x64` 子目录。本阶段只需要在已经通过完整性门禁的 aligned JPG 完整序列上重新检测68点二维landmark，不需要AU、gaze、HOG、tracked video或再次生成aligned image。冻结哈希为：

```text
FeatureExtraction.exe:
a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae

model/main_ceclm_general.txt:
7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598

readme.txt:
4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93
```

该Windows发布包默认不包含CEN patch experts。`main_ceclm_general.txt`必须同时加载以下四个二进制文件，否则会出现`Could not find CEN patch experts`和`ERROR: Could not load the landmark detector`：

```text
model\patch_experts\cen_patches_0.25_of.dat
model\patch_experts\cen_patches_0.35_of.dat
model\patch_experts\cen_patches_0.50_of.dat
model\patch_experts\cen_patches_1.00_of.dat
```

首次运行前在PowerShell执行官方下载脚本：

```powershell
$OpenFacePackage = "D:\Tools\Openface_2.2.0_win_x64\OpenFace_2.2.0_win_x64"
Set-Location $OpenFacePackage
Set-ExecutionPolicy -Scope Process Bypass
.\download_models.ps1

Get-ChildItem ".\model\patch_experts\cen_patches_*_of.dat" |
  Select-Object Name,Length,FullName
```

四个文件都存在且大小大于0后，才能重新运行项目脚本。项目脚本现在会在创建输出目录前预检这四个依赖，并在`run_manifest.json`中记录其大小和SHA-256。不要用`ccnf_*`或`svr_*`文本文件替代CEN二进制文件。

批处理脚本会在运行前强制检查 `EXACT_PASS`、OpenFace 版本、上述两个哈希和输入/输出目录；随后按视频目录名排序，逐个执行：

```text
FeatureExtraction.exe -fdir <complete_video_aligned_dir> -out_dir <output_root> -2Dfp -mloc <main_ceclm_general.txt>
```

OpenFace 2.2.0 会按文件名词典序读取 `-fdir`。当前零填充 JPG 文件名能保持帧顺序。显式指定 `-2Dfp` 后不会触发 OpenFace 的“无输出参数时输出全部特征”默认行为。不要自行附加 `-aus`、`-gaze`、`-hogalign`、`-simalign`、`-tracked`、`-verbose`、`-wild` 或 `-multi_view`。

先在Windows PowerShell中运行两个视频的门禁。`D:\Project\stc`应是从远程拉取的只读git checkout；先拉取包含本脚本修改的提交，禁止从WSL复制未提交脚本后运行正式审计。脚本会检测checkout，并核对显式commit/branch，不能产生空git provenance。

`$ImageRoot`必须直接包含300个`*_video_aligned`目录：

```powershell
$Repo = "D:\Project\stc"
Set-Location $Repo
git pull origin dev
if ($LASTEXITCODE -ne 0) { throw "git pull failed" }
if ((git status --short)) { throw "Windows checkout must be clean" }

$SourceGitCommit = (git rev-parse HEAD).Trim()
$SourceGitBranch = (git branch --show-current).Trim()
$OpenFaceRoot = "D:\Tools\Openface_2.2.0_win_x64"
$ImageRoot = "D:\Project\dataset\AVEC2014\face_images"
$OutputRoot = "D:\Project\dataset\AVEC2014\openface_aligned_landmarks_of220_a29ba49c_debug"
$IntegritySummary = "D:\Project\dataset\AVEC2014\audits\image_integrity\comparison\comparison_summary.json"

$SourceGitCommit
$SourceGitBranch

& "$Repo\scripts\run_openface_aligned_landmarks.ps1" `
  -OpenFaceRoot "$OpenFaceRoot" `
  -ImageRoot "$ImageRoot" `
  -OutputRoot "$OutputRoot" `
  -IntegrityComparisonSummary "$IntegritySummary" `
  -SourceGitCommit "$SourceGitCommit" `
  -SourceGitBranch "$SourceGitBranch" `
  -MaxVideos 2
```

debug-v2 必须满足：`video_count=2`、`pass_count=2`、`fail_count=0`、`total_images=total_csv_rows`、`success_ratio>=0.995`、`status=PASS`；`run_manifest.json` 中还必须有非空 `git_commit/git_branch`，且 `git_provenance_mode=explicit`。审计入口是：

```text
<OutputRoot>\_audit\extraction_summary.json
<OutputRoot>\_audit\video_run_summary.csv
<OutputRoot>\_audit\run_manifest.json
<OutputRoot>\_audit\logs\*.log
```

每个输入目录产生一个同名 CSV，例如 `203_1_Freeform_video_aligned.csv`，以及 OpenFace 自带的文本元数据；不会生成新的裁剪图或修改原 JPG。脚本逐视频验证 CSV 行数、`frame=1..N` 连续性、必要列、OpenFace `success` 比例，并记录 PowerShell/OpenFace/script/git/模型哈希。debug 通过后必须使用新的正式输出目录，删除 `-MaxVideos 2` 后运行全量 300 个视频：

```powershell
$OutputRoot = "D:\Project\dataset\AVEC2014\openface_aligned_landmarks_of220_a29ba49c"

& "$Repo\scripts\run_openface_aligned_landmarks.ps1" `
  -OpenFaceRoot "$OpenFaceRoot" `
  -ImageRoot "$ImageRoot" `
  -OutputRoot "$OutputRoot" `
  -IntegrityComparisonSummary "$IntegritySummary" `
  -SourceGitCommit "$SourceGitCommit" `
  -SourceGitBranch "$SourceGitBranch"
```

正式输出目录不得复用历史 `openface_features`，输出文件也不得写进任何 `*_video_aligned` JPG 子目录。进程中断后可以对同一正式目录附加 `-Resume`；脚本只跳过已经通过完整 CSV 契约的视频，缺失或不合格 CSV 会重新生成。暂不并行启动多个 OpenFace 进程，以避免 CPU/内存竞争和输出审计混乱。

### 4.2.4 AU-T0b aligned-coordinate contract

T0b 必须读取已经通过的 T0a summary/mapping 和冻结的 split 文件。推荐优先在 aligned JPG 上使用固定 OpenFace 版本重新检测 landmark，并将对应 CSV root 传给：

```bash
/home/zhen/miniconda3/bin/conda run -n light \
python scripts/audit_au_coordinate_contract.py \
  --image-root /usr/local/conda/zhen/dataset/AVEC2014/face_images \
  --openface-root /usr/local/conda/zhen/dataset/AVEC2014/openface_features \
  --frame-contract-summary logs/au_region_tracking_audit/t0a_frame_contract/tables/frame_contract_summary.csv \
  --selected-frame-mapping logs/au_region_tracking_audit/t0a_frame_contract/tables/selected_frame_mapping.csv \
  --dataset-split-file /usr/local/conda/zhen/dataset/AVEC2014/dataset_split.json \
  --aligned-openface-root /path/to/openface_aligned_landmarks \
  --mapping-method aligned_redetection \
  --output-dir logs/au_region_tracking_audit/t0b_coordinate_contract \
  --min-mapping-valid-ratio 0.995 \
  --min-in-bounds-ratio 0.80 \
  --max-overlays 120
```

若预处理阶段保存了逐帧 2x3 变换，使用 `--mapping-method explicit_affine --transform-root /path/to/transform_csv_root`；每个视频 CSV 必须包含 `frame,m00,m01,m02,m10,m11,m12`。只有存在经过确认的 aligned-space canonical template 时才使用 `--mapping-method canonical_similarity --canonical-template /path/to/template.csv`，template 格式为 `landmark_id,x,y`。

不提供上述任何合法来源时，`auto` 会输出 `BLOCKED`，不会执行检测坐标到 `112x112` 的独立比例缩放。自动检查成功仍只输出 `REVIEW_REQUIRED`；必须填写 `overlay_manifest.csv` 的 `review_status/review_notes` 并人工确认 train overlays，才能判定 T0b PASS。validation/test 不用于修改 mapping、阈值或 template。

### 4.2.5 AU-T0c 逐帧失败、原视频 presence 与曝光审计

第一步只读取已经生成的 aligned JPG 和 aligned-image OpenFace CSV，不启动 `FeatureExtraction.exe`、FaceLandmark 或任何重新检测：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_frame_recovery.py audit \
  --image-root /mnt/d/Project/dataset/AVEC2014/face_images \
  --openface-root logs/au_region_tracking_audit/openface_aligned_landmarks \
  --dataset-split-file /mnt/d/Project/dataset/AVEC2014/dataset_split.json \
  --output-dir logs/au_region_tracking_audit/frame_failure_recovery_safe_v1 \
  --exposure-safe-low-quantile 0.20 \
  --exposure-target-low-quantile 0.25 \
  --exposure-target-high-quantile 0.75 \
  --exposure-safe-high-quantile 0.80 \
  --workers 8
```

不要覆盖旧的 q10/q90 审计目录；新输出使用 `frame_failure_recovery_safe_v1`。其输出为逐帧失败表、OpenFace 连续失败块、曝光样本、视频 summary、`exposure_review_template.csv`、报告和带哈希的 run manifest。默认冻结参数为：`black_threshold=8`；纯黑 nonblack/visible ratio `<=0.01` 或 mean luma `<=1`；欠曝 sampled median `<=35`，过曝 `>=220`，异常样本比例 `>=0.5`；每视频32个等距样本。train-normal q20/q80 是处理后安全验收带，本次为 `49.018/142.171`；实际目标置于带内 q25/q75，本次为 `53.740/135.769`。欠曝计划曲线为 `log`，过曝为 `inverse_log`，曲线参数上限默认 `32`。

第二步必须回到原始视频判断人物是否在场。用户冻结的原视频目录为：

```text
D:\Project\dataset\AVEC2014\{train,dev,test}\{Freeform,Northwind}
WSL: /mnt/d/Project/dataset/AVEC2014/{train,dev,test}/{Freeform,Northwind}
```

运行全量只读 source-presence audit：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_source_presence.py \
  --dataset-root /mnt/d/Project/dataset/AVEC2014 \
  --frame-failure-manifest logs/au_region_tracking_audit/frame_failure_recovery/tables/frame_failure_manifest.csv \
  --video-failure-summary logs/au_region_tracking_audit/frame_failure_recovery/tables/video_failure_summary.csv \
  --raw-openface-root /mnt/d/Project/dataset/AVEC2014/openface_features \
  --output-dir logs/au_region_tracking_audit/source_video_presence \
  --max-run-samples 9 \
  --contact-sheet-columns 5
```

正式输出为：

```text
tables/source_video_contract.csv
tables/pure_black_source_runs.csv
tables/source_presence_review_template.csv
contact_sheets/*.jpg
reports/source_presence_report.md
run_manifest.json
```

`source_presence_review_template.csv` 中每个 pure-black run 初始为 `PENDING`。人工审阅后可把复杂 run 拆成多个不重叠 segment，但必须完整覆盖每个纯黑帧。合法决策只有：

```text
person_absent                  -> keep_invalid
person_present_detection_failure -> raw_frame_warp_only 或 keep_invalid
mixed / ambiguous             -> keep_invalid（优先继续细分）
```

禁止设置 aligned optical-flow/copy 许可。`247_3_Freeform` 的冻结示例位于 `tables/source_presence_review_247_3.csv`：`2078-2143` 人物仍在，`2144-3984` 人物缺席，`3985-4213` 人物重新进入。

全量人工 review 完成后运行强校验与报告汇总；任何 PENDING、coverage gap、重叠 segment 或非法恢复许可都会 fail closed：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/summarize_source_presence_review.py \
  --frame-failure-manifest logs/au_region_tracking_audit/frame_failure_recovery/tables/frame_failure_manifest.csv \
  --source-run-manifest logs/au_region_tracking_audit/source_video_presence/tables/pure_black_source_runs.csv \
  --review-manifest logs/au_region_tracking_audit/source_video_presence/tables/source_presence_review_full.csv \
  --output-dir logs/au_region_tracking_audit/source_video_presence/full_review
```

本次冻结结果为 38 个视频、231 个 run、4,206 帧全部 REVIEWED；2,365 帧 `raw_frame_warp_only`，1,841 帧 `keep_invalid`，无 mixed/ambiguous 残留。报告入口为 `full_review/reports/source_presence_full_review_report.md`，逐帧 gate 为 `full_review/tables/source_presence_frame_gate.csv`。

曝光审阅从下列模板开始：

```text
logs/au_region_tracking_audit/frame_failure_recovery/tables/exposure_review_template.csv
```

模板故意写为 `review_status=PENDING, review_decision=segment_review_required`，且 `observed_luma_source=sampled_audit_median_replace_before_review`，不能直接用于物化。每个候选视频必须由无间隙、无重叠的片段完整覆盖；稳定欠曝片段改为 `REVIEWED/stable_log/log`，过曝片段改为 `REVIEWED/tone_only_overexposed/inverse_log`，不恢复的片段改为 `REVIEWED/keep_raw/identity`。每个获批片段（包括整视频片段）都必须使用片段内全部可见帧重新计算 `observed_luma_median`，并把来源改为 `full_segment_visible_luma_median`；随后使用代码中的单调拟合函数冻结一个参数，使 `T(observed)=train-only target`。禁止逐帧拟合，禁止依据 BDI 或 validation/test 指标分段。

在填写人工 gate 前，先生成只读 Before/After 接触表。该命令使用审计中的时间均匀样本及 sampled luma 极值，不写正式派生帧，也不代表批准恢复：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/preview_exposure_recovery.py \
  --audit-dir logs/au_region_tracking_audit/frame_failure_recovery \
  --image-root /home/zhen/dataset/depression/avec/2014/face_images \
  --image-integrity-comparison-summary logs/au_region_tracking_audit/image_integrity/with_wsl_comparison/comparison_summary.json \
  --output-dir /tmp/avec2014_exposure_log_preview_v1 \
  --video-id 225_2_Freeform_video_aligned \
  --video-id 211_1_Freeform_video_aligned \
  --frames-per-video 12 \
  --columns 4 \
  --underexposed-target-luma 53.740102 \
  --overexposed-target-luma 135.769350 \
  --safe-low-luma 49.018280 \
  --safe-high-luma 142.171405 \
  --scan-all-frames
```

这里的 target/safe 参数仅用于在旧 q10/q90 审计输出上预览新协议，不会修改旧 manifest。由于旧 audit manifest 记录的是 `/mnt/d/.../face_images`，使用字节一致的 WSL 副本时必须显式提供 `--image-integrity-comparison-summary`。脚本只接受 `EXACT_PASS`、全量 `EXACT_MATCH`、candidate manifest SHA-256 正确且 inventory `image_root` 与当前参数一致的重定位证据。`--scan-all-frames` 对视频内全部非纯黑 aligned 帧计算指标，但接触表仍只显示代表帧。删除两个 `--video-id` 参数即可一次预览全部22个候选。

在填写最终 exposure manifest 前，先用train-only全帧IQR冻结时间稳定性阈值：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_exposure_temporal_stability.py \
  --audit-dir logs/au_region_tracking_audit/frame_failure_recovery \
  --image-root /home/zhen/dataset/depression/avec/2014/face_images \
  --image-integrity-comparison-summary logs/au_region_tracking_audit/image_integrity/with_wsl_comparison/comparison_summary.json \
  --output-dir logs/au_region_tracking_audit/exposure_temporal_stability_q90_v1 \
  --workers 8
```

`--stability-quantile`固定为`0.90`，传入其他值会fail closed。正式结果为92个train-normal参考视频，q90=`10.243687`、q95=`12.024606`；18个候选低于q90、2个处于q90-q95、2个超过q95。该命令只提供whole-video/segment-review triage，不自动生成分段；短曝光阶段仍可能被全局IQR漏掉。

完整reviewed manifest为：

```text
logs/au_region_tracking_audit/exposure_review_full_q90_v1/tables/exposure_review_manifest.csv
```

其24行完整覆盖22个候选。当前正式路由为19个`stable_log` segment、2个`tone_only_overexposed`、3个`keep_raw`；全帧变换指标见同目录`tables/exposure_review_segment_metrics.csv`和`reports/exposure_review_full_report.md`。provenance-final现已输出到`frame_failure_recovery_safe_v1`并冻结q20/q80、q25/q75及实现/输出哈希；但3帧raw-warp smoke仍未授权正式全量materialization。

在决定是否转向 raw-space 曝光校正前，运行同帧细节可恢复性审计。现有 raw/aligned OpenFace CSV 只提供68点几何对应，不比较 AU、pose 或 embedding；输入派生方案冻结后再统一版本重跑 OpenFace：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_raw_detail_recoverability.py \
  --frame-audit-dir logs/au_region_tracking_audit/frame_failure_recovery \
  --dataset-root /mnt/d/Project/dataset/AVEC2014 \
  --image-root /home/zhen/dataset/depression/avec/2014/face_images \
  --image-integrity-comparison-summary logs/au_region_tracking_audit/image_integrity/with_wsl_comparison/comparison_summary.json \
  --raw-openface-root /home/zhen/dataset/depression/avec/2014/openface_features \
  --aligned-openface-root logs/au_region_tracking_audit/openface_aligned_landmarks \
  --output-dir logs/au_region_tracking_audit/raw_detail_recoverability_v2 \
  --candidate-frames-per-video 32 \
  --reference-frames-per-video 4 \
  --contact-frames-per-video 8 \
  --calibration-quantile 0.95 \
  --min-practical-clip-advantage 0.01
```

该审计先把 raw frame 通过 RANSAC similarity transform warp 到当前 aligned 坐标，再在同一侵蚀 face hull 内比较两端剪切、q90-q10 span、robust-normalized gradient/Laplacian/entropy。正式结果为92个 train-normal 视频/365个有效参考帧、22候选701帧/685个有效比较；median transform inlier ratio `0.941`、RMSE `0.687 px`。加入1个百分点最小实际效应后 `raw_detail_recoverable=0`：16视频为 `tone_only_or_keep_raw`，6视频为 `inconclusive_keep_raw_until_review`。因此不实现 raw-space exposure correction；raw-frame warp 只保留给人物在场但 aligned 纯黑的另一类失败帧。

当前`materialize`同时强制要求已完成的source-presence与exposure review manifest。它只会对已批准片段应用固定log-family曲线；3帧smoke尚未接入正式materialize，因此所有纯黑帧仍保持invalid，其中人物仍在的帧记录为`raw_frame_warp_required_but_not_implemented`：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_frame_recovery.py materialize \
  --audit-dir logs/au_region_tracking_audit/frame_failure_recovery_safe_v1 \
  --image-root /mnt/d/Project/dataset/AVEC2014/face_images \
  --source-review-manifest /path/to/completed_source_presence_review.csv \
  --exposure-review-manifest /path/to/completed_exposure_review.csv \
  --output-root /tmp/avec2014_exposure_review \
  --layout sparse_overlay \
  --video-id 225_2_Freeform_video_aligned \
  --video-id 211_1_Freeform_video_aligned
```

`sparse_overlay` 只包含改变帧，不能单独作为当前 dataset 的 `IMAGE_DIR`。只有 source presence、曝光和未来 raw-frame warp overlay 全部通过后，才允许使用独立完整 mirror；命令同样必须提供 review manifest：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_frame_recovery.py materialize \
  --audit-dir logs/au_region_tracking_audit/frame_failure_recovery_safe_v1 \
  --image-root /mnt/d/Project/dataset/AVEC2014/face_images \
  --source-review-manifest /path/to/completed_source_presence_review.csv \
  --exposure-review-manifest /path/to/completed_exposure_review.csv \
  --output-root /mnt/d/Project/dataset/AVEC2014/face_images_repaired_v1 \
  --layout mirror \
  --link-mode hardlink
```

`hardlink`要求源和派生目录位于同一文件系统；不满足时显式改为`copy`或`symlink`。正式output root必须不存在或为空，且不能位于原始`face_images`内。当前不要再次运行FaceLandmark。

3帧raw-frame warp smoke使用WSL本地、经`EXACT_PASS`证明与审计根逐字节一致的aligned JPG；只有原始MP4从`/mnt/d`读取：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_frame_recovery.py raw-warp-smoke \
  --audit-dir logs/au_region_tracking_audit/frame_failure_recovery_safe_v1 \
  --dataset-root /mnt/d/Project/dataset/AVEC2014 \
  --image-root /home/zhen/dataset/depression/avec/2014/face_images \
  --image-integrity-comparison-summary logs/au_region_tracking_audit/image_integrity/with_wsl_comparison/comparison_summary.json \
  --source-presence-gate logs/au_region_tracking_audit/source_video_presence/full_review/tables/source_presence_frame_gate.csv \
  --raw-openface-root /home/zhen/dataset/depression/avec/2014/openface_features \
  --aligned-openface-root logs/au_region_tracking_audit/openface_aligned_landmarks \
  --output-dir logs/au_region_tracking_audit/raw_frame_warp_smoke_v3 \
  --max-frames 3 \
  --split train
```

默认只接受train、1-3帧、完整source-presence gate、前后锚点距离`<=8`、tracking valid ratio`>=0.60`、transform inlier ratio`>=0.50`、RMSE`<=2.50 px`、tracked-vs-interpolated transform分歧`<=5 px`和warped landmark in-bounds ratio`>=0.80`。输出为`AUTO_PASS_REVIEW_REQUIRED`，只写独立`derived/`、contact sheet、逐帧表、报告和带全部输入/实现/输出哈希的manifest；禁止用前后aligned脸合成目标帧，禁止把smoke目录作为训练`IMAGE_DIR`。

### 4.2.6 Validity-aware temporal slicing 只读审计

该命令只合并既有manifest并生成候选切片，不读取BDI/prediction、不改图片、不运行OpenFace：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_validity_aware_slicing.py \
  --dataset-split-file /home/zhen/dataset/depression/avec/2014/dataset_split.json \
  --video-failure-summary logs/au_region_tracking_audit/frame_failure_recovery/tables/video_failure_summary.csv \
  --frame-failure-manifest logs/au_region_tracking_audit/frame_failure_recovery/tables/frame_failure_manifest.csv \
  --source-presence-gate logs/au_region_tracking_audit/source_video_presence/full_review/tables/source_presence_frame_gate.csv \
  --output-dir logs/au_region_tracking_audit/validity_aware_slicing_v1 \
  --sample-step 10 \
  --max-seq-len 2000 \
  --window-frames 600 1200 2000 \
  --min-clip-frames 600
```

正式判读优先看 `tables/slicing_aggregate_summary.csv` 和 `reports/validity_aware_slicing_report.md`。当前2000帧结果只表示pure-black/presence/OpenFace-success容量估计；它没有量化大遮挡、大偏转和严重出界，不能直接作为训练clip。`2000`主窗口和“每视频随机一个窗口”均未冻结。完整新协议见 `VALIDITY_AWARE_TEMPORAL_SLICING_PLAN.md`。

### 4.2.7 FACE-S1 人脸可用性phase-1分布审计

优先使用WSL本地JPG，避免跨文件系统访问。该根目录已通过`EXACT_PASS`证明与原审计源逐字节一致：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_face_usability.py \
  --dataset-split-file /home/zhen/dataset/depression/avec/2014/dataset_split.json \
  --image-root /home/zhen/dataset/depression/avec/2014/face_images \
  --aligned-landmark-root logs/au_region_tracking_audit/openface_aligned_landmarks \
  --frame-audit-dir logs/au_region_tracking_audit/frame_failure_recovery_safe_v1 \
  --source-presence-gate logs/au_region_tracking_audit/source_video_presence/full_review/tables/source_presence_frame_gate.csv \
  --exposure-review-manifest logs/au_region_tracking_audit/exposure_review_full_q90_v1/tables/exposure_review_manifest.csv \
  --image-integrity-comparison-summary logs/au_region_tracking_audit/image_integrity/with_wsl_comparison/comparison_summary.json \
  --canonical-sample-step 30 \
  --contact-frames-per-reason 12 \
  --contact-sheet-columns 4 \
  --output-dir logs/au_region_tracking_audit/face_usability_phase1_v2
```

该命令只输出confidence、landmark in-frame、face hull coverage、正深度landmark-derived yaw/pitch/roll、PnP重投影误差、blur、transform residual和jump的train/val/test分布及train-only contact sheets，不自动批准阈值。现有完整运行耗时约26分钟。

jump必须使用相邻帧成对复核：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/audit_face_usability_temporal_review.py \
  --source-run-dir logs/au_region_tracking_audit/face_usability_phase1_v2 \
  --aligned-landmark-root logs/au_region_tracking_audit/openface_aligned_landmarks \
  --output-dir logs/au_region_tracking_audit/face_usability_temporal_review_v1 \
  --columns 4
```

随后生成去重的global/local-geometry/boundary双通道人工复核模板：

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/prepare_face_usability_threshold_review.py \
  --source-run-dir logs/au_region_tracking_audit/face_usability_phase1_v2 \
  --temporal-review-dir logs/au_region_tracking_audit/face_usability_temporal_review_v1 \
  --output-dir logs/au_region_tracking_audit/face_usability_threshold_review_v2
```

正式模板为`tables/face_usability_threshold_review_template.csv`。同一帧分别填写global face、local geometry和temporal boundary标签；可见landmark失败允许global可用但local geometry不可用。local geometry通过只表示可进入后续四区polygon/margin overlay审计，不批准具体local crop。三个review-status字段必须全部改为`REVIEWED`，且标签只能使用instructions报告列出的闭集。当前模板124帧全部PENDING，不能直接传给FACE-S2。

WSL本地还存在历史OpenFace特征副本`/home/zhen/dataset/depression/avec/2014/openface_features`。FACE-S1不读取它；后续外部pose/quality/AU弱标签审计如需使用，必须先说明所需CSV字段、工具版本和与当前split/frame contract的一致性门禁。

人工复核后第二轮才允许传入冻结threshold manifest，生成：

```text
tables/face_usability_frame_manifest.csv
tables/face_usable_run_manifest.csv
tables/face_clip_candidate_manifest.csv
tables/face_quality_exclusion_summary.csv
reports/face_usability_report.md
run_manifest.json
```

候选clip统计必须覆盖`window=300/600/1200/2000`和`overlap=0/25/50%`，报告每video/subject/task clip分布。任何输出都不能把clip数称为独立subject样本数。详细字段见`SHORTCUT_AUDIT_DESIGN.md`第15节，原始视频问题见`AVEC2014_SOURCE_DATA_QUALITY.md`。

### 4.3 OpenFace alignment geometry audit

```bash
python scripts/audit_alignment_geometry.py \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/alignment_geometry \
  --frame-width 640 \
  --frame-height 480 \
  --sample-step 1
```

> 注意：当前 OpenFace CSV 坐标系约为 640x480，不是 aligned 112x112。

### 4.4 Black artifact audit

```bash
python scripts/audit_black_artifacts.py \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/black_artifacts \
  --sample-step 10
```

### 4.5 Shortcut audit（OpenFace quality + shortcut predictor）

```bash
python scripts/audit_shortcuts.py \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/shortcut_audit
```

### 4.6 Embedding identity retrieval audit

**单 split 默认布局：**

```bash
python scripts/audit_identity_retrieval.py \
  --features-npz <LOG_DIR>/default/mtl_lite/version_0/diagnostics/embeddings/test_features.npz \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/identity_retrieval \
  --top-k 5
```

**多 split 布局：**

```bash
python scripts/audit_identity_retrieval.py \
  --features-npz <LOG_DIR>/default/mtl_lite/version_0/diagnostics/test/embeddings/test_features.npz \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/test/identity_retrieval \
  --top-k 5
```

### 4.7 Severity calibration verification

需要先在 val split 上导出预测：

```bash
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/mtl_lite/version_0 \
  --ckpt best \
  --split val test
```

然后运行校准审计：

```bash
python scripts/audit_severity_calibration.py \
  --val-predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/val/regression/val_predictions.csv \
  --test-predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/severity_calibration
```

### 4.8 Task inconsistency mixed-factor audit

```bash
python scripts/audit_task_inconsistency.py \
  --predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/test/regression/test_predictions.csv \
  --black-artifacts-summary <LOG_DIR>/default/mtl_lite/version_0/diagnostics/black_artifacts/tables/black_artifact_summary.csv \
  --openface-quality-summary <LOG_DIR>/default/mtl_lite/version_0/diagnostics/shortcut_audit/tables/openface_quality_summary.csv \
  --alignment-geometry-summary <LOG_DIR>/default/mtl_lite/version_0/diagnostics/alignment_geometry/tables/alignment_geometry_summary.csv \
  --temporal-sampling-summary <LOG_DIR>/default/mtl_lite/version_0/diagnostics/temporal_sampling/tables/temporal_sampling_summary.csv \
  --output-dir <LOG_DIR>/default/mtl_lite/version_0/diagnostics/task_inconsistency \
  --top-n 20
```

所有 `--*-summary` 参数都是可选的；缺少的变量将不会进入相关分析。

---

## 5. 跨 run 汇总入口

### 5.1 Prediction runs summary

```bash
python scripts/summarize_prediction_runs.py \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --baseline rgb \
  --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run border_black_feather=<LOG_DIR>/default/border_black_feather/version_0/diagnostics/regression/test_predictions.csv
```

`PATH` 可以是 CSV 文件，也可以是包含 `test_predictions.csv` 的目录。使用 `--split val` 可汇总验证集预测。

输出：

```text
analysis_outputs/rgb_input_ablation_summary/
  tables/prediction_run_summary.csv
  tables/severity_bias_summary.csv
  tables/task_consistency_summary.csv
  tables/pairwise_baseline_improvement.csv
  reports/prediction_runs_report.md
```

### 5.2 Training overfit summary

```bash
python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=<LOG_DIR>/default/rgb/version_0/metrics.csv \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/metrics.csv \
  --run behavior=<LOG_DIR>/default/behavior_baseline/version_0/metrics.csv
```

### 5.3 Identity retrieval multi-run summary

```bash
python scripts/summarize_identity_retrieval_runs.py \
  --output-dir analysis_outputs/identity_retrieval_summary \
  --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/identity_retrieval \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/identity_retrieval \
  --run border_black_feather=<LOG_DIR>/default/border_black_feather/version_0/diagnostics/identity_retrieval
```

### 5.4 Severity calibration multi-run summary

```bash
python scripts/summarize_severity_calibration_runs.py \
  --output-dir analysis_outputs/severity_calibration_summary \
  --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/severity_calibration \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/severity_calibration \
  --run border_black_feather=<LOG_DIR>/default/border_black_feather/version_0/diagnostics/severity_calibration
```

### 5.5 Mechanism summary（合并预测 / identity / calibration）

```bash
python scripts/summarize_mechanism.py \
  --prediction-summary analysis_outputs/rgb_input_ablation_summary/tables/prediction_run_summary.csv \
  --severity-bias-summary analysis_outputs/rgb_input_ablation_summary/tables/severity_bias_summary.csv \
  --task-consistency-summary analysis_outputs/rgb_input_ablation_summary/tables/task_consistency_summary.csv \
  --identity-summary analysis_outputs/identity_retrieval_summary/tables/identity_retrieval_run_summary.csv \
  --calibration-summary analysis_outputs/severity_calibration_summary/tables/severity_calibration_run_summary.csv \
  --output-dir analysis_outputs/mechanism_summary
```

---

## 6. RGB vs behavior 预测比较

```bash
python scripts/compare_behavior_predictions.py \
  --rgb-predictions <LOG_DIR>/default/mtl_lite/version_0/diagnostics/regression/test_predictions.csv \
  --behavior-predictions <LOG_DIR>/default/behavior_baseline/version_0/diagnostics/behavior/test_predictions.csv \
  --output-dir analysis_outputs/rgb_behavior_comparison
```

---

## 7. 完整实验工作流示例

### 7.1 跑一组 RGB 输入消融并汇总

```bash
# 1. 训练 rgb baseline
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/rgb.yaml

# 2. 训练 center_mask
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/center_mask.yaml

# 2b. 训练 central_face_mask：覆盖眼、鼻、嘴和主要脸颊的新遮挡对照
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/central_face_mask.yaml

# 3. 训练边界平滑消融
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/edge_soften_only.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/border_blur_fill.yaml

# 3b. 训练身份抑制与组合消融（Identity x Boundary 2x2）
python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/identity_texture_suppressed.yaml

python scripts/train_mtl_lite.py \
  --override configs/regression_only_baseline.yaml \
  --override configs/input_ablation/identity_texture_suppressed_edge_soften.yaml

# 4. 对 run 生成 test 诊断（单 split 默认即可）
python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/rgb/version_0 --ckpt best

python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/center_mask/version_0 --ckpt best

python scripts/diagnose_mtl_lite.py \
  --run-dir <LOG_DIR>/default/rgb_ablation_central_face_mask/version_0 --ckpt best

# 5. 汇总比较（包含 Identity x Boundary 2x2 四组）
python scripts/summarize_prediction_runs.py \
  --output-dir analysis_outputs/rgb_vs_center_mask \
  --baseline rgb \
  --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run central_face_mask=<LOG_DIR>/default/rgb_ablation_central_face_mask/version_0/diagnostics/regression/test_predictions.csv \
  --run edge_soften_only=<LOG_DIR>/default/edge_soften_only/version_0/diagnostics/regression/test_predictions.csv \
  --run border_blur_fill=<LOG_DIR>/default/border_blur_fill/version_0/diagnostics/regression/test_predictions.csv \
  --run identity_texture_suppressed=<LOG_DIR>/default/identity_texture_suppressed/version_0/diagnostics/regression/test_predictions.csv \
  --run identity_texture_suppressed_edge_soften=<LOG_DIR>/default/identity_texture_suppressed_edge_soften/version_0/diagnostics/regression/test_predictions.csv
```

### 7.2 完整多因素审计工作流

```bash
RUN_DIR=<LOG_DIR>/default/mtl_lite/version_0

# 1. 导出 val + test 预测
python scripts/diagnose_mtl_lite.py --run-dir $RUN_DIR --ckpt best --split val test

# 2. 各项审计
python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/split_integrity

python scripts/audit_temporal_sampling.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir $RUN_DIR/diagnostics/test/temporal_sampling \
  --sample-step 10 --max-seq-len 2000 --sampling-strategy stride_head

python scripts/audit_alignment_geometry.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir $RUN_DIR/diagnostics/test/alignment_geometry \
  --frame-width 640 --frame-height 480 --sample-step 1

python scripts/audit_black_artifacts.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --image-root /path/to/AVEC2014/face_images \
  --output-dir $RUN_DIR/diagnostics/test/black_artifacts \
  --sample-step 10

python scripts/audit_shortcuts.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir $RUN_DIR/diagnostics/test/shortcut_audit

python scripts/audit_identity_retrieval.py \
  --features-npz $RUN_DIR/diagnostics/test/embeddings/test_features.npz \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/identity_retrieval \
  --top-k 5

python scripts/audit_severity_calibration.py \
  --val-predictions $RUN_DIR/diagnostics/val/regression/val_predictions.csv \
  --test-predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --output-dir $RUN_DIR/diagnostics/test/severity_calibration

python scripts/audit_task_inconsistency.py \
  --predictions $RUN_DIR/diagnostics/test/regression/test_predictions.csv \
  --black-artifacts-summary $RUN_DIR/diagnostics/test/black_artifacts/tables/black_artifact_summary.csv \
  --openface-quality-summary $RUN_DIR/diagnostics/test/shortcut_audit/tables/openface_quality_summary.csv \
  --alignment-geometry-summary $RUN_DIR/diagnostics/test/alignment_geometry/tables/alignment_geometry_summary.csv \
  --temporal-sampling-summary $RUN_DIR/diagnostics/test/temporal_sampling/tables/temporal_sampling_summary.csv \
  --output-dir $RUN_DIR/diagnostics/test/task_inconsistency \
  --top-n 20
```

---

## 8. 输出目录速查表

| 脚本 | 主要输出目录 | 关键产物 |
|---|---|---|
| `train_mtl_lite.py` | `experiment/<group>/<name>/version_N/` | `metrics.csv`、`checkpoints/`、`resolved_config.yaml` |
| `train_behavior_baseline.py` | `experiment/<group>/<name>/version_N/diagnostics/behavior/` | `val_predictions.csv`、`test_predictions.csv` |
| `diagnose_mtl_lite.py`（单 split） | `<run_dir>/diagnostics/` | `regression/test_predictions.csv`、`embeddings/test_features.npz` |
| `diagnose_mtl_lite.py`（多 split） | `<run_dir>/diagnostics/<split>/` | `<split>/regression/<split>_predictions.csv`、`<split>/embeddings/<split>_features.npz` |
| `audit_split_integrity.py` | `<output_dir>/` | `tables/split_integrity_report.md` 等 |
| `audit_temporal_sampling.py` | `<output_dir>/` | `tables/temporal_sampling_summary.csv`、报告 |
| `audit_alignment_geometry.py` | `<output_dir>/` | `tables/alignment_geometry_summary.csv`、报告 |
| `audit_black_artifacts.py` | `<output_dir>/` | `tables/black_artifact_summary.csv`、报告 |
| `audit_shortcuts.py` | `<output_dir>/` | `tables/openface_quality_summary.csv`、`shortcut_audit_report.md` |
| `audit_identity_retrieval.py` | `<output_dir>/` | `tables/embedding_identity_summary.csv`、相似度图、报告 |
| `audit_severity_calibration.py` | `<output_dir>/` | `tables/severity_calibration_fit.csv`、测试摘要、报告 |
| `audit_task_inconsistency.py` | `<output_dir>/` | `tables/task_inconsistency_manifest.csv`、`task_artifact_correlation.csv`、报告 |
| `summarize_prediction_runs.py` | `<output_dir>/` | `prediction_run_summary.csv`、`severity_bias_summary.csv`、报告 |
| `summarize_training_overfit.py` | `<output_dir>/` | `training_overfit_summary.csv`、报告 |
| `summarize_identity_retrieval_runs.py` | `<output_dir>/` | `identity_retrieval_run_summary.csv`、报告 |
| `summarize_severity_calibration_runs.py` | `<output_dir>/` | `severity_calibration_run_summary.csv`、报告 |
| `summarize_mechanism.py` | `<output_dir>/` | `mechanism_summary.csv`、`mechanism_report.md` |
| `audit_layerwise_identity_probe.py` | `<output_dir>/` | `test_layerwise_features.npz`、`tables/layerwise_identity_summary.csv`、报告 |
| `audit_error_identity_coupling.py` | `<output_dir>/` | `tables/error_identity_correlation.csv`、`high_error_high_identity_cases.csv`、报告 |
| `audit_artifact_weaklabels.py` | `<output_dir>/` | `tables/artifact_weaklabel_summary.csv`、`tables/artifact_weaklabel_correlation.csv`、报告 |
| `summarize_artifact_weaklabels_matched.py` | `<output_dir>/` 或原 A3 目录 | `tables/artifact_weaklabel_summary_matched.csv`、`tables/artifact_weaklabel_correlation_matched.csv`、报告 |
| `summarize_severity_imbalance.py` | `<output_dir>/` | `tables/severity_imbalance_summary.csv`、报告 |

---

## 9. Shortcut Stage A 证据收口

Stage A 是当前 task-nuisance 主线的前置证据，必须在实现 identity-adversarial baseline 或 `TaskNuisanceBlock` 前完成。四个诊断回答四个问题：

```text
A1 身份信息在哪些层可分？           -> identity attacker 接入层 / 后续扩展证据
A2 预测错误是否与身份相似性耦合？   -> identity-adversarial 是强抑制还是仅监控
A3 OpenFace artifact 是否与误差耦合？-> shortcut/artifact probes 与 group-wise evaluation
A4 分数段不均是否驱动 minimal/severe bias？-> severity-balanced 是 Stage B 基线还是 Stage D 支线
```

核心逻辑约束：**A1 只证"embedding 有身份"，A2 才证"prediction 用了身份"**。只有 A1+A2 同时成立，才进入 Stage B 的强 identity suppression。

所有 Stage A 脚本均为离线诊断：只读 checkpoint + 数据，不污染 val/test，不改训练 forward/loss/checkpoint。详细脚本/输出规格见 `docs/SHORTCUT_AUDIT_DESIGN.md` 第 13 节。

### 9.1 前置：确保 run 已生成基础诊断产物

Stage A 的 A2/A3/A4 依赖已有的 `test_predictions.csv` 和 P0 审计 summary。先对 RGB baseline run 跑标准诊断与审计：

```bash
RUN_DIR=<LOG_DIR>/default/rgb/version_0

# 1. 导出 test 预测 + 最终 pooled embedding（A1 的 layer_shared 对照基线来自这里）
python scripts/diagnose_mtl_lite.py \
  --run-dir $RUN_DIR \
  --ckpt best \
  --split test

# 2. P0 审计（A3 整合这些 summary，A4 整合 prediction/severity/calibration summary）
python scripts/audit_black_artifacts.py \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --image-root <IMAGE_ROOT> \
  --output-dir $RUN_DIR/diagnostics/black_artifacts \
  --sample-step 10

python scripts/audit_alignment_geometry.py \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --openface-root <OPENFACE_CSV_ROOT> \
  --output-dir $RUN_DIR/diagnostics/alignment_geometry \
  --frame-width 640 --frame-height 480

python scripts/audit_shortcuts.py \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --openface-root <OPENFACE_CSV_ROOT> \
  --output-dir $RUN_DIR/diagnostics/shortcut_audit

python scripts/audit_temporal_sampling.py \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --image-root <IMAGE_ROOT> \
  --output-dir $RUN_DIR/diagnostics/temporal_sampling \
  --sample-step 10 --max-seq-len 2000

# 3. severity calibration（A4 需要 delta_ccc）
python scripts/audit_severity_calibration.py \
  --run-dir $RUN_DIR \
  --ckpt best \
  --output-dir $RUN_DIR/diagnostics/severity_calibration
```

### 9.2 A1 Layer-wise Identity Probe

逐层导出 embedding，复用 `compute_identity_retrieval_metrics` 做同 subject 检索。定位身份信息来自 backbone 哪一层。

```bash
python scripts/audit_layerwise_identity_probe.py \
  --run-dir $RUN_DIR \
  --ckpt best \
  --split test \
  --output-dir $RUN_DIR/diagnostics/layerwise_identity \
  --top-k 5
```

可选参数：

- `--layers layer_stem,layer_block_3,layer_block_6,layer_block_9,layer_block_11,layer_shared`：只跑指定层（默认全部 `LAYERWISE_PROBE_LAYERS`）。
- `--device cuda` / `--batch-size 4`：诊断设备与 batch。
- `--predictions $RUN_DIR/diagnostics/regression/test_predictions.csv`：可选，用于 metadata enrichment。

输出：

```text
test_layerwise_features.npz                    # 多键 NPZ（features_<layer_name>）
tables/layerwise_identity_summary.csv          # 每层 same_subject_top1/3/5、paired_rank、severity/task agree
tables/layerwise_identity_per_query.csv        # 逐 query 逐层（A2 输入）
reports/layerwise_identity_report.md
```

注意：backbone 若不暴露 `patch_embed`/`blocks`（如 CNN），脚本会跳过无法解析的层并打印 `[LAYER-PROBE] Skipped unresolved layers`，不会硬失败。

### 9.3 A2 Prediction Error x Identity Similarity Coupling

证明预测"使用了"身份，而非仅 embedding"包含"身份。

```bash
python scripts/audit_error_identity_coupling.py \
  --features-npz $RUN_DIR/diagnostics/layerwise_identity/test_layerwise_features.npz \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --per-query-csv $RUN_DIR/diagnostics/layerwise_identity/tables/layerwise_identity_per_query.csv \
  --output-dir $RUN_DIR/diagnostics/error_identity_coupling \
  --max-cases 20
```

可选参数：

- `--per-query-csv`：可选。提供后额外计算 paired_rank / severity_agree 的秩相关；不提供则只报连续 identity similarity 的 Pearson 相关。
- `--layers layer_shared`：只分析指定层。

输出：

```text
tables/error_identity_correlation.csv          # 每层 corr(|err|,id_sim)、corr(res,rank) 等
tables/severity_bin_identity_error_summary.csv # 按 minimal/mild/moderate/severe 分段
tables/high_error_high_identity_cases.csv      # coupling_score 最高的 case（论文 case-study anchor）
reports/identity_error_coupling_report.md
```

判读：`corr(|err|, id_sim) > 0` 表示高身份相似伴随大误差 → 预测用了身份；耦合集中在 severe bin → 支持"severe 低估与身份记忆耦合"；相关≈0 → 身份仅在 embedding，Stage B 转为风险监控。

### 9.4 A3 Artifact Weak-label Audit (evaluation / probe)

整合 4 个 P0 审计 summary，决定 artifact/quality/context 变量的审计、probe、case-study 和分组评估用途。当前第一版不默认建立 `z_art` 训练分支。

```bash
python scripts/audit_artifact_weaklabels.py \
  --predictions $RUN_DIR/diagnostics/regression/test_predictions.csv \
  --black-artifacts $RUN_DIR/diagnostics/black_artifacts/tables/black_artifact_summary.csv \
  --alignment-geometry $RUN_DIR/diagnostics/alignment_geometry/tables/alignment_geometry_summary.csv \
  --openface-quality $RUN_DIR/diagnostics/shortcut_audit/tables/openface_quality_summary.csv \
  --temporal-sampling $RUN_DIR/diagnostics/temporal_sampling/tables/temporal_sampling_summary.csv \
  --output-dir $RUN_DIR/diagnostics/artifact_weaklabels \
  --coupling-threshold 0.2
```

可选：4 个 source CSV 任一缺失都能跑（只整合提供的）。`--coupling-threshold` 控制 |corr(abs_error)| 阈值（默认 0.2）。

输出：

```text
tables/artifact_weaklabel_summary.csv          # 每视频弱标签 join 预测误差
tables/artifact_weaklabel_correlation.csv      # 每弱标签与 4 个目标的相关（按 |corr(abs_error)| 降序）
reports/artifact_weaklabel_report.md           # 含 artifact/quality 变量定位
```

判读（报告自动给出）：|corr(abs_error)| ≥ 阈值 → 纳入 shortcut/artifact probe、case study 和 group-wise evaluation；只与 true_bdi 耦合 → 采集偏置，只作为 label-confound 证据；无弱标签过阈值 → 仅审计出口。

若 `artifact_weaklabel_summary.csv` 同时包含全数据集 OpenFace weak labels 和当前 split 的 prediction rows，需要额外生成 matched-only correlation。该步骤只保留 `pred_bdi`、`residual`、`abs_error` 非空的视频，避免 OpenFace-only 行把 `n` 显示成全量 300：

```bash
python scripts/summarize_artifact_weaklabels_matched.py \
  --summary val=$RUN_DIR/diagnostics/val/artifact_weaklabels/tables/artifact_weaklabel_summary.csv \
  --summary test=$RUN_DIR/diagnostics/test/artifact_weaklabels/tables/artifact_weaklabel_summary.csv \
  --output-dir $RUN_DIR/diagnostics/artifact_weaklabels_matched
```

若使用独立 `analysis_outputs` 汇总目录，可采用当前项目的组织方式：

```bash
python scripts/summarize_artifact_weaklabels_matched.py \
  --summary val=logs/analysis_outputs/val/artifact_weaklabels/tables/artifact_weaklabel_summary.csv \
  --summary test=logs/analysis_outputs/test/artifact_weaklabels/tables/artifact_weaklabel_summary.csv \
  --output-dir logs/analysis_outputs/artifact_weaklabels_matched
```

后续 Stage A3 结论、论文表格、shortcut/artifact probe 变量筛选和 artifact-risk group 构造应使用：

```text
logs/analysis_outputs/artifact_weaklabels_matched/val/tables/artifact_weaklabel_correlation_matched.csv
logs/analysis_outputs/artifact_weaklabels_matched/test/tables/artifact_weaklabel_correlation_matched.csv
```

原始 `artifact_weaklabel_correlation.csv` 只作为弱标签整合中间产物，不作为最终预测误差耦合口径。

如果只处理单个 split，且希望输出写回原 A3 目录，可省略 `--output-dir`：

```bash
python scripts/summarize_artifact_weaklabels_matched.py \
  --summary $RUN_DIR/diagnostics/test/artifact_weaklabels/tables/artifact_weaklabel_summary.csv
```

### 9.5 A4 Severity Imbalance / Prediction Compression Summary

决定 severity-balanced regression 是 Stage B 必跑基线还是 Stage D 支线。

先准备跨 run summary（若尚未生成）：

```bash
# prediction + severity bias summary
python scripts/summarize_prediction_runs.py \
  --baseline rgb \
  --output-dir analysis_outputs/rgb_input_ablation_summary \
  --run rgb=$RUN_DIR/diagnostics/regression/test_predictions.csv

# severity calibration multi-run summary（A4 需要 delta_ccc）
python scripts/summarize_severity_calibration_runs.py \
  --output-dir analysis_outputs/severity_calibration_summary \
  --run rgb=$RUN_DIR/diagnostics/severity_calibration
```

再跑 A4：

```bash
python scripts/summarize_severity_imbalance.py \
  --prediction-summary analysis_outputs/rgb_input_ablation_summary/tables/prediction_run_summary.csv \
  --severity-bias analysis_outputs/rgb_input_ablation_summary/tables/severity_bias_summary.csv \
  --calibration-summary analysis_outputs/severity_calibration_summary/tables/severity_calibration_run_summary.csv \
  --output-dir analysis_outputs/severity_imbalance_summary
```

可选：`--run rgb --run center_mask` 显式指定包含的 run（默认取三源并集）。

输出：

```text
tables/severity_imbalance_summary.csv          # 每 run 的 imbalance_ratio、compression、minimal/severe residual、推荐
reports/severity_imbalance_report.md           # 含 stage_b_baseline / stage_d_side_branch / stage_d_optional 三档
```

判读：`imbalance_ratio ≥ 2.0` + minimal 高估 + severe 低估 + calibration 不改善 → `stage_b_baseline`（E2 必跑）；否则降为 Stage D 支线或 optional。

### 9.6 Stage A 收口

A1-A4 运行完毕后，在 `CURRENT_STATUS.md` 和 `RGB_OVERFITTING_AUDIT_PLAN.md` 写出四个结论并关闭 Stage A：

```text
1. 身份存在（A1）：身份信息主要来自哪一层，强度如何
2. 身份参与预测（A2）：身份相似性是否与预测误差/偏置耦合
3. 伪迹参与错误（A3）：哪些 artifact/quality 弱标签进入 probe、case study 或 group-wise evaluation
4. severity 失衡（A4）：severity-balanced regression 是 Stage B 必跑还是 Stage D 支线
```

只有 1+2 同时成立才进入 Stage B 的 identity-adversarial 强抑制路径；若只有 1，identity 仅作风险监控。该证据不直接授权 Stage C 第一版加入 `z_id`。关闭后不再扩展普通输入滤镜、黑边替换、灰度/模糊/mask 族。

---

## 10. 常见问题

**Q: 训练输出没有生成 `version_0` 而是覆盖到了旧目录？**

A: 检查 `configs/local_paths.yaml` 是否正确设置了 `LOG_DIR`。新 runner 使用 `<LOG_DIR>/<EXPERIMENT_GROUP>/<EXPERIMENT_NAME>/version_N/`；确认 override 中 `EXPERIMENT_NAME` 与预期一致。

**Q: `diagnose_mtl_lite.py` 报错找不到 `test_predictions.csv`？**

A: 如果使用 `--split val test`，预测文件在 `<run_dir>/diagnostics/test/regression/test_predictions.csv`；如果单 split 默认，文件在 `<run_dir>/diagnostics/regression/test_predictions.csv`。

**Q: behavior baseline 报错 `DATASET.OPENFACE_ROOT is required`？**

A: 必须额外提供一个 override YAML 或在 `configs/local_paths.yaml` 中设置 `DATASET.OPENFACE_ROOT`。

**Q: 旧端到端训练（`scripts/train.py`）还能用吗？**

A: 当前旧端到端模型 `src/models/end_to_end.py` 缺失，legacy 训练入口不可用。如需复现旧结果，需先恢复 legacy 模块或显式要求修复。

**Q: 配置加载失败？**

A: 确认 `configs/local_paths.yaml` 存在，或运行训练脚本时加上 `--allow-missing-local-paths`。但真实数据训练必须提供 `IMAGE_DIR`、`LABEL_DIR`、`DATASET_SPLIT_FILE`。
