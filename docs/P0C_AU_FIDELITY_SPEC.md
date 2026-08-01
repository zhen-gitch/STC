# PB-P0C Core AU Physical Fidelity Specification

## 1. Status and authorization boundary

Implementation package: `CODE-20260731-PB-P0C-CORE-AU-FIDELITY-v1`.

This package implements the frozen policy, a raw-video OpenFace extraction entry point, a read-only fidelity audit, its CLI, tests, and this specification. It does not run OpenFace, generate raw-reference data, run the formal audit, modify a dataset/model/runner/training config, commit, push, or authorize P0D/P0E/training.

The first version evaluates only the currently authorized core group:

```text
AU12_r / AU14_r / AU15_r
```

`AU4/6/7/10/17` remain outside the source contract. Their physical presence in a 182-column CSV does not authorize value access. Pose/head and gaze are excluded from P0C eligibility.

## 2. Immutable evidence bindings

| Evidence | Required SHA-256 |
|---|---|
| P0C fidelity policy v1 | `2f605261958c0cb7f82c8f871c33c05c81e12c146f73ddd0b4b25d13e9276d2d` |
| dataset split | `979dc7c5ea622a6b4575bf1d6f89393bf9563fe8f3cdc5bec80fff73787e2e24` |
| source-video contract | `12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31` |
| aligned rich v2 run manifest | `95bc6f08638c6b2ee381aa8725108aab3f259b3883d5b893f9a30abe74b16aba` |
| P0B run manifest | `e2dfda24fe653d5b318077faa1dc83b2549be802c40c2d7d1a452b2196a6d1ae` |
| P0B selected-target manifest | `028ce8c11715fc48019d255c8d1aaf0d38d282ad896c582ccbf44997b23ee076` |
| `FeatureExtraction.exe` | `a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae` |
| OpenFace model | `7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598` |
| OpenFace AU predictor manifest | `b65b923db38e75ee53ea7f85d614882f2f834571c257c7e0c3637d348a0192d6` |

The expected full source is exactly 300 videos and 493,141 frames. A formal audit accepts only a full, structurally passing, clean-checkout raw-reference manifest. The raw and aligned runs must also bind identical complete OpenFace binary and model inventory manifests, not only the executable and top-level model hashes. Debug output cannot be relabeled as full evidence.

## 3. Ownership and data flow

```text
source_video_contract.raw_video_path
 -> FeatureExtraction -f raw.mp4 -2Dfp -pose -aus
 -> new immutable raw CSV root + extraction manifests

raw CSV + aligned-rich v2 CSV + split + P0B PASS + frozen policy
 -> exact video_id/task/frame join
 -> joint raw/aligned quality mask
 -> video metrics
 -> equal-weight task median inside NNN_M subject
 -> subject median inside physical split
 -> train-only AU/group decision
 -> val/test report-only tables
```

The extractor owns raw-video path validation, OpenFace binary/model/profile provenance, complete row/schema/hash checks, immutable output creation, and an end-of-run check that the script, source contract, git state, binaries, and model files did not change during extraction. The Python audit owns source binding, exact pairing, masks, statistics, decisions, and reports. Dataset, sampler, model, trainer, evaluator, labels, predictions, and checkpoints are not involved.

OpenFace emits pose columns because the raw reference uses the same feature profile as aligned rich v2. The audit projects only `frame`, `success`, `confidence`, and the three core AU values. Pose/gaze/extension-AU value access counts must remain zero. The historical `raw_openface_csv` source-contract field is never dereferenced or opened.

## 4. Match and mask contract

For source `s in {raw, aligned}` and AU `a`:

```text
quality_s(t) = success_s(t) == 1 and confidence_s(t) >= 0.8
matched_a(t) = quality_raw(t) and quality_aligned(t)
               and raw_a(t), aligned_a(t) are finite and within [0, 5]
pair_a(t) = matched_a(t-1) and matched_a(t) and frame(t)=frame(t-1)+1
```

Both CSVs must contain exactly `frame=1..N`, with one row per source frame and the same video/task identity. Offset, interpolation, imputation, pseudo-zero, smoothing, and differencing across an invalid gap are forbidden. Non-finite or out-of-range selected values are technical blockers rather than values silently inserted into a metric.

## 5. Frozen statistics

Statistics are computed first per task video, then per `NNN_M` subject by the median of estimable task metrics, then across subjects by the median. Freeform and Northwind are never concatenated and do not count as independent subjects. A subject with one estimable task remains reportable as `single_task_only`.

### 5.1 Spearman

- Use joint-valid intensity samples and average ranks for ties.
- A video requires at least 120 matched frames and at least three unique values on each side.
- An AU requires at least 30 estimable train subjects and at least 20 estimable videos per task.
- PASS requires train subject median rho `>=0.70`.
- Use a 10,000-sample subject bootstrap, seed 42, percentile 95% CI; the lower bound must be strictly `>0.50`.

### 5.2 Significant-change direction

- Only exact consecutive joint-valid pairs are used.
- A raw reference event satisfies `abs(delta_raw)>=0.10`.
- A match requires the same sign and `abs(delta_aligned)>=0.05`.
- A video requires at least 20 reference events; an AU requires at least 20 estimable train subjects.
- PASS requires the train subject median direction rate `>=0.75`.
- Aligned-only event rate is reported but is not a v1 gate.

### 5.3 Amplitude

- Per-video amplitude is `Q95-Q05` on the same joint-valid values.
- Raw amplitude below 0.10 is not estimable; aligned amplitude zero yields a ratio of zero.
- An AU requires at least 30 estimable train subjects and at least 20 estimable videos per task.
- PASS requires median `aligned_amplitude/raw_amplitude` within inclusive `[0.5,2.0]`.

### 5.4 Lag

- `rho(k)=Spearman(raw[t], aligned[t+k])`, with positive lag meaning aligned follows raw.
- Search `k=-5..5`; every lag uses at least 120 valid pairs and three unique values per side.
- The strict argmax and ties are always reported.
- Canonical lag is zero when `rho(0) >= max_k rho(k)-0.02`; otherwise it is the deterministic strict peak.
- PASS requires canonical lag zero, at least 30 subjects at every lag, and at least 20 estimable videos per task.

### 5.5 Cross-talk

The audit reports subject-median `raw_i -> aligned_j`, `raw_i -> raw_j`, and `aligned_i -> aligned_j` 3x3 matrices. Cross-talk is `REPORT_ONLY`; v1 has no threshold because natural AU co-activation makes an unregistered off-diagonal cutoff uninterpretable.

## 6. Machine decision

Technical and scientific states are separate:

- `audit_status=BLOCKED`: provenance, hash, schema, video set, frame join, selected value domain, or completeness failed. Every AU is `NOT_EVALUATED`.
- `audit_status=PASS` with AU `BLOCKED_EVIDENCE`: computation completed but the registered minimum evidence was not met.
- AU `FAIL_METRIC`: evidence is sufficient, but one or more registered metric gates failed.
- AU `PASS`: all five registered gates passed.

The core group is `ELIGIBLE` only when all three AUs pass. Any metric failure makes it `INELIGIBLE_METRIC`; evidence deficiency makes it `BLOCKED_INCOMPLETE_EVIDENCE`. A failed AU cannot be silently removed while retaining the same group/version name.

Every decision fixes:

```text
training_authorized=false
p0d_authorized=false
p0e_authorized=false
```

## 7. Files and outputs

Implementation:

- `configs/behavior_alignment/privileged_behavior_au_fidelity_policy_v1.json`
- `src/diagnostics/privileged_behavior_au_fidelity.py`
- `scripts/audit_privileged_behavior_au_fidelity.py`
- `scripts/run_openface_raw_au_fidelity_reference.ps1`

Formal audit output is a new directory that must not equal, contain, or be contained by either OpenFace input root. It contains:

```text
tables/au_fidelity_coverage.csv
tables/au_fidelity_video_metrics.csv
tables/au_fidelity_subject_metrics.csv
tables/au_fidelity_lag_curve.csv
tables/au_fidelity_cross_talk.csv
tables/au_fidelity_group_decision.csv
tables/au_fidelity_issues.csv
au_fidelity_decision.json
reports/au_fidelity_report.md
run_manifest.json
```

`run_manifest.json` records command, branch/commit/dirty state, all input and implementation hashes, output hashes, CPU/float64 metric precision, access counts, and the explicit no-model/no-training boundary.

### 7.1 Command templates

The raw-reference command writes a new dataset-side output root and runs OpenFace in the foreground. It must not be executed until its separate debug/full run package is authorized:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\Project\stc\scripts\run_openface_raw_au_fidelity_reference.ps1 `
  -OpenFaceRoot D:\Tools\Openface_2.2.0_win_x64 `
  -RawVideoRoot D:\Project\dataset\AVEC2014 `
  -SourceVideoContract D:\Project\stc\logs\au_region_tracking_audit\source_video_presence\tables\source_video_contract.csv `
  -OutputRoot D:\Project\dataset\AVEC2014\openface_raw_au_fidelity_of220_a29ba49c_debug2_v1 `
  -SourceGitCommit <authorized-commit> `
  -SourceGitBranch dev `
  -MaxVideos 2
```

The later full command uses a new `..._full_v1` root and omits `-MaxVideos`; it refuses a dirty checkout. The formal audit is read-only over all source roots and writes only its new output directory:

```bash
/home/zhen/miniconda3/envs/light/bin/python \
  scripts/audit_privileged_behavior_au_fidelity.py \
  --policy configs/behavior_alignment/privileged_behavior_au_fidelity_policy_v1.json \
  --policy-sha256 2f605261958c0cb7f82c8f871c33c05c81e12c146f73ddd0b4b25d13e9276d2d \
  --dataset-split-file /home/zhen/dataset/depression/avec/2014/dataset_split.json \
  --source-video-contract logs/au_region_tracking_audit/source_video_presence/tables/source_video_contract.csv \
  --aligned-openface-root /mnt/d/Project/dataset/AVEC2014/openface_aligned_behavior_of220_a29ba49c_v2 \
  --aligned-run-manifest /mnt/d/Project/dataset/AVEC2014/openface_aligned_behavior_of220_a29ba49c_v2/_audit/run_manifest.json \
  --raw-openface-root /mnt/d/Project/dataset/AVEC2014/<authorized-full-raw-root> \
  --raw-run-manifest /mnt/d/Project/dataset/AVEC2014/<authorized-full-raw-root>/_audit/run_manifest.json \
  --raw-run-manifest-sha256 <reviewed-full-raw-manifest-sha256> \
  --p0b-run-manifest logs/privileged_behavior_alignment/p0b_core_coverage_full_a29ba49c_v1/run_manifest.json \
  --p0b-selected-target-manifest logs/privileged_behavior_alignment/p0b_core_coverage_full_a29ba49c_v1/selected_target_manifest.json \
  --output-dir logs/privileged_behavior_alignment/<authorized-p0c-output-v1>
```

## 8. Separately authorized run sequence

1. Commit/push and synchronize a clean Windows checkout are separate actions.
2. `RUN-20260731-PB-P0C-RAWREF-DEBUG2-v1`: deterministic first two source-contract videos, new debug root.
3. `RUN-20260731-PB-P0C-RAWREF-FULL-v1`: clean checkout, new full root, expected 3-6 hours CPU.
4. `RUN-20260731-PB-P0C-AUDIT-v1`: read-only Python audit, expected 5-20 minutes CPU.
5. Result-document synchronization, commit, and push remain separate packages.

Failed or interrupted roots are preserved as evidence. They cannot be overwritten, resumed, or promoted; a rerun needs a new versioned output directory.

## 9. Remaining interpretation risks

- Raw OpenFace is not AU ground truth. P0C measures representation stability, not facial-action accuracy or depression relevance.
- Raw `-f` and aligned `-fdir` are different OpenFace input modes; binary/model/profile matching controls most provenance but does not remove every tracker/decoder difference.
- Joint-valid masking can induce selection bias when raw and aligned detection failures differ; one-sided and joint coverage are therefore reported.
- AU intensities are zero-inflated and temporally autocorrelated. Subject clustering avoids treating frames/tasks as independent, but sparse AUs may still be blocked for insufficient evidence.
- Five conjunctive gates and all-members group eligibility are deliberately conservative and may reject a usable component.
