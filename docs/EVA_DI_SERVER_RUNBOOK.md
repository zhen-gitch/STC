# EVA-DI 服务器运行手册（RIB-STD-v1）

结构对齐 trans/ribformer 的服务器运行标准（`docs/CODE_AND_RESOURCE_RUN_STANDARD.md`、
`EVA_THROUGHPUT_SERVER_RUNBOOK.md`、`D092_MAINLINE_SERVER_SEQUENCE.md`）：命令一律从工作副本
根目录执行；机器路径全部经 `EVA_DI_*` 环境注入（等价 `RIB_PATHS_OVERRIDE` 机制）；大体积中间
产物写 run 根不写源码树；每次 run 自动留存 raw/resolved config + sha256 + git provenance。

## §0 初始化（每台机器一次）

```bash
cd /usr/local/conda/zhen/project/stc                     # 服务器工作副本根（对齐 rib-former 根约定）
git fetch origin && git checkout dev && git pull --ff-only
git rev-parse HEAD                                        # 记下 commit，写入实验记录

cp configs/eva_di/paths.server.example.yaml configs/eva_di/paths.server.yaml   # gitignored，机器私有
# 按本机实况编辑 paths.server.yaml（数据只读根 + 可写 run 根）

set -a; . <(python - <<'PY'
import yaml
p = yaml.safe_load(open("configs/eva_di/paths.server.yaml"))["paths"]
m = {"of3_root":"OF3_ROOT","avec_root":"AVEC_ROOT","split_file":"SPLIT_FILE",
     "label_dir":"LABEL_DIR","weight_path":"WEIGHT","cache_root":"CACHE_ROOT",
     "log_root":"LOG_ROOT","output_root":"OUTPUT_ROOT",
     "behavior_norm_root":"NORM_ROOT"}
import shlex
[print(f"EVA_DI_{v}={shlex.quote(str(p[k]))}") for k,v in m.items() if p.get(k)]
PY
); set +a

export PYTHONPATH=.:src
export MPLCONFIGDIR=/tmp/matplotlib-eva_di
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1            # 权重离线钉死（对齐 rib）
export CUDA_VISIBLE_DEVICES=<空闲卡号>
bash scripts/eva_di/eva_di_run.sh env-check               # fail-closed：缺根/缺权重/盘满当场拒跑
```

权重钉定：`eva02_small_patch14_224.mim_in22k`，`model.safetensors`，SHA256
`d3d632352efbd0a0a8269dce114ab8a214833f85ce6f738860221f11ab0f0c2f`（装载前逐字节校验，失败拒跑）。
离线分发件走 rib_transfer 同风格 bundle（tar + `.sha256` sidecar）。

## §1 数据与特征缓存（二选一）

- **A. 复用本地缓存 bundle（快，sha 一致性即证明）**：取
  `eva_di_cache_uniform512_v1_*.tar.zst(+.sha256)` → 校验 → 解到 `$EVA_DI_CACHE_ROOT`；
  然后跑 oracle 等价门（见 B 末）。
- **B. 服务器重抽**：`bash scripts/eva_di/eva_di_run.sh extract`（EVA02 全量 206 录制，
  4090 约 30–40 分钟），**随后必跑** `bash scripts/eva_di/eva_di_run.sh oracle`——
  与 trans ribformer 管线逐位等价门，非零退出 = 停。

## §2 正式矩阵（新 REGNORM 协议：plain MSE + 归一化目标 + grad_audit + CCC/PCC/MAE/RMSE）

```bash
bash scripts/eva_di/eva_di_run.sh plan        # dry-run：臂/状态/命令清单，不启动
bash scripts/eva_di/eva_di_run.sh launch      # 顺序执行 stage1，首败即停
bash scripts/eva_di/eva_di_run.sh viz         # 全部 run 的训练曲线 + 跨臂对比
bash scripts/eva_di/eva_di_run.sh report      # matrix_summary.{md,csv}（含身份攻击列）
```

- stage1 臂：REF / T1 / T3 / FULL / DERM + 消融 EVA-ONLY、OF3-ONLY（`configs/eva_di/matrix_v1.yaml`，
  协议变更继承 base：`reg_weighting: plain`、epochs 100、patience 18）。
- λ 扩展臂（本地 tt2/tt4 已探明方向）：按 `di_full.yaml` 派生 `lam: 0.3/1.0/3.0` 的
  derived yaml 放 run 根（同 trial 模式），`matrix.py --arms ... --seeds ...` 消费；
  planner 不会伪造 derived config，缺文件即 blocked。
- 探索臂按 rib worktree 惯例隔离：
  `git worktree add /usr/local/conda/zhen/project/stc-<exp> <commit>`。
- 新协议本地锚点（勿与旧加权协议混比）：REF CCC **0.4999**@57，FULL λ1.0 CCC **0.5314**@88
  （100ep 未早停——服务器建议 `training.epochs_max: 150` 起步）。
- GRL 审计默认 `on_grl_violation: report`：λ 小步长下 GRU 反放的跨前向噪声只标记不停训
  （机理见 docs/EVA_DI_AUDIT_20260905.md 与 grad_audit 模块注释）。

## §3 验收门（每臂）

1. `run_provenance.json`：git commit/branch/dirty、config raw+resolved sha256、split/manifest sha、
   device/precision、seed 全齐（trainer 自动写）。
2. `summary.json`：`best_defined=true`、`grad_audit.replay_bitexact_all=true`；
   `grad_audit.violation_note` 若存在须在实验记录中判读。
3. `identity/identity_metrics.json`：p_mean/v_h0 双出口 top1/top3/pair-AUROC/A1，coverage=1.0。
4. test split 全程锁定；best 只按 val_ccc；sealed 产物与普通输出根不同父不互嵌（对齐 rib §109）。

## §4 回收

```bash
bash scripts/eva_di/eva_di_run.sh archive      # run 根打包 tar.zst + sha256（rib_transfer 同风格）
# 本地：解包后 rsync 到 stc/logs/eva_di_runs_<date>/（gitignored），复跑 viz/matrix_report 亦可
```

## §5 边界

本手册不含：多 seed 正式矩阵的启动授权、commit/push、任何写数据集根的操作。
停止条件：oracle 门非零 / env-check fail / 双臂以上 grad_audit FLAG。
