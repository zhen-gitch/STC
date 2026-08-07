# EXPERIMENT_LOG.md

> 文档职责：按时间追加实验和实现记录，用于追溯证据。当前权威结论见 `CURRENT_STATUS.md` / `RGB_OVERFITTING_AUDIT_PLAN.md`；文档导航见 `DOCS_GUIDE.md`。

This log records completed project maintenance, smoke validation, and experiment
workflow milestones. Keep entries concise and reproducible.

## 2026-08-05

### AU-guided landmark-localized RGB route

- Approved planning route `PLAN-20260805-AU-GUIDED-LANDMARK-RGB-v1` and synchronized it under docs-only package `DOC-20260805-AU-GUIDED-LANDMARK-RGB-v1`.
- Clarified the main hypothesis: AU/FACS defines eye-brow, nose-cheek, and mouth/lower-face semantics; aligned-space 68-point landmarks determine geometry, containment, stability, and view validity; the model learns only global/local RGB.
- Closed P0B as technically complete and preserved P0C as a valid negative result: the mask-aware audit passed technically, but AU12/14/15 were `INELIGIBLE_METRIC`. This blocks current AU numerical supervision, not semantic region cropping.
- Removed extension-A/B AU numerical qualification, PB-P0D/P0E, AU heads, AU loss, and the old `GLA-FULL` from the current execution chain. They remain historical or require a separately registered future branch.
- Registered the new sequence: region contract code -> physical-train geometry pilot -> policy freeze -> full region audit -> default-off four-view dataset/model -> smoke -> `GLA-C-REF` versus `GLA-RGB-FULL` -> region/grid controls -> paired multi-seed -> locked benchmark.
- No code, config, schema, data, experiment, commit, or push was authorized by this documentation package.

## 2026-07-30

### DOC-20260730-GLA-HEADLESS-PHOTOAUG-v2 documentation freeze

- Authorized scope was documentation only. The current GLA candidate is now headless and AU-only: global face plus eye-brow, nose-cheek and mouth-lower-face views use one shared backbone, shared projection, global-residual validity-aware fusion, one-layer GRU, masked temporal mean and BDI regression. Head motion and gaze are excluded; pose/head extraction and coverage remain historical audit evidence.
- Auxiliary supervision is limited to core AU12/14/15, extension A AU4/6/7 and extension B AU10/17, including cross-region AU6 and AU14. The frozen loss is `L_BDI + ramp * lambda_AU * L_AU_fixed_denominator`; AU gradients end at the shared backbone/projection and AU heads rather than entering fusion, GRU or the BDI head.
- Deterministic P1/P2 tone normalization, color normalization and exposure-derived mirrors no longer form the current GLA input route. Physical-train online augmentation keeps spatial transforms shared across views/time, while exposure and ordinary ColorJitter may be sampled independently by view and remain fixed over that view's complete sequence. Validation/test/inference remain unaugmented; padding and invalid pixels remain unchanged.
- The subtraction ladder is now `S0 GLA-FULL -> S1 GLA-S1-NO-AU-B -> S2 GLA-S2-CORE-AU -> S3 GLA-RGB-FULL -> S4 GLA-C-REF`; `GLA-FULL-NO-EXPOSURE-AUG` is a matched strategy control. Seed42 is capped at 12 full fits.
- No code, config, dataset, split, label, generated data or generated run/log artifact was changed; no smoke, training, commit or push was run. Future P0B code, P0 runs, GLA implementation, smoke, training, commit and push each remain separately gated by named disclosure and explicit authorization.

### PB-P0A3 authoritative full-rich v2 and full coverage decision

- Authoritative source: `/mnt/d/Project/dataset/AVEC2014/openface_aligned_behavior_of220_a29ba49c_v2`. It completed 300/300 videos, 493,141/493,141 rows, one 182-column schema, 587 MB and 10,064.6 seconds. It produced no HOG, tracked video or regenerated aligned images; content/summary/run-manifest hash bindings pass.
- Source provenance is clean Windows `dev@2538248`; extractor SHA-256 is `64375c6575066619cb21ce94d213e7457446df961a8d9565bd554eee68db3cc5`; source-contract SHA-256 is `12f50c2311b9d89dde81e27fa83891226ca5f447ed7d3d56fe38cf673a1c5a31`.
- The incomplete v1 source is prohibited from resume, reuse or P0B. Its 282 complete/1 incomplete/17 missing videos and absent final manifests remain failure evidence only.
- Authoritative A2 output: `logs/privileged_behavior_alignment/p0a2_mask_aware_policy_full_a29ba49c_v2/`. Result: `PASS_FULL_SOURCE_COVERAGE`, 0 blockers, 0 warnings, 486,441 quality/AU-valid frames and 485,863 head-valid adjacent pairs. Physical-train overall/Freeform/Northwind coverage is 0.989273/0.983944/0.996984, task gap is approximately 0.01304, and the lowest train video has 180 AU-valid frames and 179 head pairs.
- Legacy extraction summary remains expected `FAIL` at 259 strict PASS / 41 strict FAIL and 486,640/493,141 OpenFace successes. It is diagnostic only under the frozen mask-aware policy. Val/test are report-only; P0B and training remain unauthorized.
- Known report bug: the full PASS decision still writes `next_action=STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES`. Known compatibility risk: current P0 core may still require legacy extraction/content rows to be strict PASS. Both require a separately disclosed and authorized code package before P0B; hash/schema/join/provenance/value-domain gates must remain unchanged.

### Full-model-first hybrid ablation strategy and agent disclosure gate (initial plan; head axis superseded)

- This initial plan included dependency-aware subtraction `-HEAD -> -AU10/17 -> -AU4/6/7 -> -all AU -> -locals`. The earlier head axis is retained here only as decision history and is superseded by the headless S0--S4 contract recorded above.
- Grid, no-cross and subject-deranged shuffled-aux controls remain mandatory. A component absent from P0E is `SKIPPED_INELIGIBLE`, not an ablation result. `GLA-FULL` is unrelated to the legacy full model and Stage C `C-FULL`.
- This entry is docs/governance only. No dataset/model/runner/config was changed and no P0B, smoke, training, benchmark, commit or push was authorized or run.
- Root `AGENTS.md` now requires a named implementation package disclosure before every code/config/test/data/run/commit/push package. The agent must report boundaries, architecture/data flow, interfaces, split/test access, exact commands/resources/outputs, validation, risks and sub-agent roles, end the turn, and wait for explicit authorization of that package.

## 2026-07-29

### PB-P0A2 versioned mask-aware source/coverage policy

- Added frozen label-blind policy `configs/behavior_alignment/behavior_source_coverage_policy_v1.json` (SHA-256 `9a3fb739078ab36e1fa4f8f67a8c8b167a76329941a591ad0ef9dd0c731a1926`), the read-only validator/CLI and focused tests. No BDI label, prediction, checkpoint, model or training input is exposed.
- Source completeness remains hard-fail across selected video set, rows, schema, hashes, provenance, forbidden outputs and input contracts. Frame validity is independently recomputed as `success == 1 && confidence >= 0.8`; legacy strict `0.995` status is retained as diagnostic only. A legacy FAIL in `csv_content_manifest.csv` is accepted only when it matches `video_run_summary.csv`; hashes, rows, size, schema and non-coverage extractor issues remain blocking.
- Authoritative run: `logs/privileged_behavior_alignment/p0a2_mask_aware_policy_pilot20_2538248_v5/`. Result: `PASS_PILOT_MASK_AWARE_FEASIBILITY`, 20 videos, 25,980 frames, 25,440 quality/AU-valid frames, 25,410 adjacent valid head pairs, 0 blockers and 3 future-full warnings.
- The warnings are physical-train pilot-subset overall `0.965171 < 0.98`, Freeform `0.932505 < 0.95`, and task gap `0.067495 > 0.05`. Because the pilot contains only 8 physical-train videos, aggregate thresholds are `REPORT_ONLY_INCOMPLETE_SPLIT`; they must be enforced on the complete train split after a separately authorized full extraction.
- Decision fields remain `full_rich_authorized=false`, `p0b_authorized=false`, and `training_authorized=false`. The next candidate action is review of the three risks and separate PB-P0A3 authorization; no full extraction or training was started.
- Validation: `7 passed` for the focused policy tests and `64 passed` for the pure-Python PB contract/P0/policy regression set.

## 2026-07-28

### PB-P0 privileged AU/head data-contract infrastructure and blocked landmark-only audit

- Authorization was limited to PB-P0 source/data/schema/coverage/train-only-normalization infrastructure and the landmark-only blocked audit; it did not include a full 300-video rich extraction. Added the fail-closed contract core and full-dataset landmark-only audit, the CLI `scripts/audit_privileged_behavior_contract.py`, and the independent no-gaze extractor `scripts/run_openface_aligned_behavior_features.ps1`. No dataset/model/runner, auxiliary head, training config, smoke training or PB-P1 implementation was added.
- Frozen targets are `AU12_r/AU14_r/AU15_r` and `d_pose_Rx_dt/d_pose_Ry_dt/d_pose_Rz_dt`. Gaze is absent from extractor arguments, targets, normalization and loss. Historical raw-video rich OpenFace files are explicitly excluded.
- The initial formal landmark-only run is `logs/privileged_behavior_alignment/p0_contract_landmark_only_blocked_v1/`. It completed exact structural joins for `300/300` split videos and `493141` aligned JPG frames, but the source schema lacks `AU12_r/AU14_r/AU15_r/pose_Rx/pose_Ry/pose_Rz`; rich core coverage is therefore `0/300`, train-only target statistics are unavailable and the result is correctly `BLOCKED` rather than silently falling back to historical features.
- The audit records gaze target/value/normalizer/loss access counts as zero, raw OpenFace feature-file open/value access counts as zero, and val/test normalization access as zero. Its outputs include frame contract, group coverage, target statistics, source-provenance fidelity, explicit `NOT_RUN_P0_PLACEHOLDER` identity/task-risk rows, issue table, selected-target manifest, report and run manifest. Neither identity/task risk nor aligned-pose physical fidelity was run or passed.

### PB-P0 provenance hardening and authoritative landmark-only v2

- `logs/privileged_behavior_alignment/p0_contract_landmark_only_blocked_v2/` supersedes v1 as the current authoritative landmark-only audit; v1 is retained only as historical evidence.
- The v2 result remains intentionally `BLOCKED`: `300` videos, `493141` frames, `0` core-audited videos, `0` rich-schema videos, `300` exact joins and `301` blocking issues (`300 × missing_behavior_columns`, `1 × train_only_statistics_unavailable`). All nine outputs and all eight recorded output hashes were independently verified.
- The implementation SHA-256 values frozen by the v2 manifest are `34a72712c113a3d387d7898b3bf969a2a40962b5e61fe4fcbdc21de7e2151e02` for P0 orchestration, `6027c587df39c748e8572cbe2426d1ea0749fe171a9b02c9b63b337528862227` for the contract core and `f86793ae0d4ecf938c6fc3e17ae09c3f8cc24e104ff64dc2b81ef7f5d63cf5a3` for the CLI.
- Strict-rich provenance now requires the exact reviewed extractor SHA and frozen OpenFace 2.2.0 package/profile, a valid commit/branch with available and clean source git status, `run_scope=full_dataset`, `max_videos=0`, exact 300-video/493141-frame scope and an exactly frozen `min_success_ratio=0.995`.
- The extractor rejects row-width, parsing and value-domain violations: exact 182-column rows, sequential frames, zero image-directory timestamps, finite binary success, confidence in `[0,1]`, AU intensities in `[0,5]` and rotations in `[-pi,pi]`.
- The final run manifest hashes the adjacent PASS extraction summary. Both that summary and the run manifest hash the same `csv_content_manifest.csv`; PB-P0 rechecks each CSV's rows, schema, byte size and SHA-256.
- For an eligible full-rich source, PB-P0 also rechecks every current aligned JPG against the frozen candidate image manifest by exact relative path, byte size and SHA-256. This expensive gate is `NOT_APPLICABLE`, not passed, for the current landmark-only v2 source.
- Validation passed with `65 passed in 47.43s` for `tests/test_privileged_behavior_contract.py`, `tests/test_privileged_behavior_p0.py` and `tests/test_openface_aligned_behavior_features_script.py`, including the real Windows PowerShell 5 contract tests.
- The eight-file PB-P0 core was frozen on branch `dev` at commit `2685fa49459d6e79499849c2927fce17a2ddc1f8` (`2685fa4`, `feat: add privileged behavior P0 audit contract`). PB status documentation is recorded in the independent docs-only follow-up created on top of that commit; the validated core commit was not amended.
- The current P0 machine status vocabulary is `PASS/BLOCKED`; its train-statistics default only rejects exactly constant (`std == 0`) targets, and its coverage table is reporting-only. The generic `selected_target_manifest.next_gate` remains a historical P0 field and must not override the later `Windows sync -> fresh debug2 -> pilot20 -> policy` route.
- No full 300-video rich extraction, dataset/model/runner change, auxiliary head, PB-P1/PB-P2 implementation or training was authorized or performed. Gaze remains excluded.

### PB-P0 aligned-JPG rich extractor two-video debug

- Final debug output: `/mnt/d/Project/dataset/AVEC2014/openface_aligned_behavior_of220_debug2c_20260728`.
- OpenFace 2.2.0 ran with the fixed `quality_2d_pose_au_no_gaze_no_hog_v1` profile (`-2Dfp -pose -aus`). Both videos passed: `2/2 PASS`, `1920/1920` image/CSV rows, `1920` successes, one 182-column schema, and zero gaze, HOG, tracked-video or regenerated-aligned outputs.
- The core contract check found 1,920 valid AU rows and 1,918 valid head-velocity rows; the first valid frame of each video has no predecessor and is correctly masked for velocity. These are two-video schema/mask checks, not physical-fidelity evidence.
- OpenFace `-fdir` timestamps were all zero. Head dynamics therefore use the frozen `source_video_contract.csv` rather than CSV timestamps: `300/300 PASS`, aligned/raw frame counts equal, `fps=30`, and `t=(source_frame_id-1)/fps` with wrapped angular differences on adjacent valid frames only.
- This two-video debug validates the executable/schema/provenance path only. It is not a full rich extraction and cannot authorize PB-P1. The next gates are `2685fa4`-based Windows clean sync, a current-script fresh debug2, pilot20, and a label-blind threshold/coverage policy decision. A new 300-video rich output is conditional on all four gates and separate authorization; it must then pass PB-P0 core, aligned-pose physical fidelity and final-descriptor identity/task/exposure/quality risk before any training starts.

## 2026-07-19

### FACE-S1 and LM-T0b AI-assisted review copies

- Preserved both authoritative `PENDING` templates and generated separate review copies with reviewer `Codex-AI-assisted`, review date, per-row notes and explicit `human_countersigned=false` provenance.
- FACE-S1 reviewed all 124 train frames and 372 decision cells. Global labels are `111 usable / 8 out-of-frame / 3 major-occlusion / 2 blur`; local labels are `92 candidate / 12 landmark-missing / 7 out-of-frame / 6 unstable / 5 extreme-pose / 2 other`; all 12 paired jumps and all 124 temporal labels are `no_boundary`. Independent validation checked canonical row order, immutable fields, closed label sets, 124 current images, 12 previous images and 11 contact sheets. Completed/validated CSV SHA-256 is `e7a09000da2781ea12d0496abd62722957cb3d10c469797a3a217d0c0c6cb802`.
- LM-T0b reviewed all 120 train overlays as `115 PASS / 5 UNCERTAIN / 0 FAIL`. The uncertain rows are `T0B-0007`, `T0B-0031`, `T0B-0035`, `T0B-0115` and `T0B-0119`, all due to occlusion or severe crop. The repository validator checked immutable fields and overlay hashes; completed/validated CSV SHA-256 is `575541d57d510f16b44310cb233a90ab54ed22fa017edabdf5e5b36ec901a8ac`.
- The LM-T0b overlay status is `REVIEW_REQUIRED`; the source `40 FAIL / 260 REVIEW_REQUIRED` keeps `coordinate_contract_status=FAIL` and `authorized=false`. FACE-S1 remains `AI_ASSISTED_REVIEW_COMPLETE_HUMAN_COUNTERSIGN_PENDING`. No threshold manifest, `face_usable` approval, FACE-S2 clip, local crop or training authorization was produced.

Validation commands:

```bash
/home/zhen/miniconda3/envs/light/bin/python scripts/validate_au_coordinate_overlay_review.py \
  --review-package-dir logs/au_region_tracking_audit/t0b_overlay_review_of220_v2 \
  --completed-review logs/au_region_tracking_audit/t0b_overlay_review_ai_assisted_20260719/tables/overlay_review_completed_codex_ai.csv \
  --output-dir logs/au_region_tracking_audit/t0b_overlay_review_validation_ai_assisted_20260719

/home/zhen/miniconda3/envs/light/bin/python scripts/validate_face_usability_threshold_review.py \
  --review-package-dir logs/au_region_tracking_audit/face_usability_threshold_review_v2 \
  --completed-review logs/au_region_tracking_audit/face_usability_threshold_review_ai_assisted_20260719/tables/face_usability_threshold_review_completed_codex_ai.csv \
  --output-dir logs/au_region_tracking_audit/face_usability_threshold_review_validation_ai_assisted_20260719_v4 \
  --review-kind ai_assisted \
  --review-date 2026-07-19
```

## 2026-07-18

### LM-T0b full aligned-coordinate contract audit

- Completed the full read-only coordinate audit with the frozen OpenFace 2.2.0 aligned-image landmark run `t0b_coordinate_contract_of220_v1`.
- The run covers 300 videos and 41,016 model-selected frames. Its decision is `260 REVIEW_REQUIRED / 40 FAIL / 0 BLOCKED`; the 40 FAIL videos are train/val/test=`12/13/15` and all fail the frozen `mapping_valid_ratio >= 0.995` gate.
- The aligned OpenFace extraction itself contains 493,141 rows and 486,640 `success=1` frames (`0.986817` overall), matching the independent frame-failure audit. T0b additionally records 364 sampled `mapping_source_detection_failed` frames and 3 low in-bounds frames in `222_1_Freeform_video`.
- The severe examples are `238_3_Freeform_video` (mapping-valid ratio `0.0000`), `207_2_Freeform_video` (`0.703704`) and `242_1_Northwind_video` (`0.769841`). These are not repaired by interpolation or by changing thresholds.
- The authoritative report and tables are under `logs/au_region_tracking_audit/t0b_coordinate_contract_of220_v1/`. The 120 train overlays remain blank for manual `review_status/review_notes`; dynamic masks, local crops and training remain blocked. The historical placeholder directory `t0b_coordinate_contract` is not used.

### LM-T0b train-overlay review package and validator

- Added a read-only review-package generator that verifies train-only rows and source artifacts, hashes every overlay, and writes 12 paginated contact sheets plus a separate PENDING review template. The original T0b overlay manifest is not modified.
- Added a fail-closed signed-review validator. Immutable frame/path/hash/mapping fields must match the template; every row requires `REVIEWED`, `PASS|FAIL|UNCERTAIN`, reviewer and date, with notes for non-PASS decisions.
- The validator reports overlay-review and source-coordinate decisions separately. Even an all-PASS 120-overlay review cannot override the 40 automated source-video FAIL decisions or authorize local crops.
- Generated the authoritative `t0b_overlay_review_of220_v2` package: 120 overlays, 30 per selection reason, 12 contact-sheet pages. The earlier v1 package was generated before the validator was added and is superseded by v2.
- A model-assisted visual precheck found no systematic scale/translation error. Hand occlusion, severe crop and locally obscured anatomy remain explicit human decisions, including `T0B-0007`, `T0B-0031`, `T0B-0035`, `T0B-0115` and `T0B-0119`.

### Exposure-derived OpenFace audit-feature extraction implementation

- Added `scripts/run_openface_exposure_features.ps1` as a separate Windows pipeline for formally materialized exposure-processed aligned JPGs. It explicitly requests quality, pose, gaze, 2D/3D landmarks, PDM parameters and AU outputs while disabling HOG and aligned-image generation.
- Added fail-closed gates for a colocated `COMPLETE + mirror` materialization manifest, full source/derived relative-frame parity, repair manifest hashes, per-modified-frame source/derived SHA-256, exposure-only repair types, git/review provenance, frozen OpenFace/CEN/AU models, CSV row/frame/schema parity and contract-aware resume.
- Strengthened future `materialization_manifest.json` outputs with completion status, repair/failure CSV hashes and an explicit complete-tree/relative-path/hash frame contract. Historical manifests are not silently upgraded.
- Kept the AU boundary unchanged: extracted AU values are for paired input diagnostics only and are not authorized as RGB-model inputs, auxiliary targets or training supervision.
- Validation completed locally with focused pytest (`12 passed`), Python compile checks and `git diff --check`. Native PowerShell parser execution was unavailable in the current sandbox, so Windows parser/debug execution remains part of the two-video gate.

### Raw-vs-exposure OpenFace paired audit implementation

- Added a read-only frame-paired diagnostic and CLI that require matching OpenFace core/CEN provenance, exact video/frame joins and the exposure run's recorded materialization SHA-256.
- Added per-video success transitions, confidence deltas, 68-point displacement and temporal motion metrics, plus per-feature 3D/pose/gaze/PDM/AU drift only when both schemas and rich-model hashes match.
- Kept landmark-only references valid for the common quality/2D subset while explicitly refusing to fill unavailable rich fields from historical raw-video OpenFace outputs.
- Outputs are descriptive and remain `REVIEW_REQUIRED`; they do not access BDI labels/predictions or claim exposure normalization improves downstream utility.
- Synthetic coverage includes landmark-only pairing, full rich-feature pairing, provenance rejection and a read-only CLI contract.

## 2026-07-17

### Frozen OpenFace 2.2.0 release-package path correction

- Changed the Windows extraction root to
  `D:\Tools\Openface_2.2.0_win_x64`. The actual executable package is the
  nested `OpenFace_2.2.0_win_x64` directory rather than the old
  `x64\Release` source-build layout.
- Froze the new package hashes: `FeatureExtraction.exe=a29ba49c...96ae`,
  `main_ceclm_general.txt=7efbef33...3598`, and
  `readme.txt=4ccdd65f...1d93`. These differ from the historical package and
  must not be mixed in one landmark version.
- Updated the PowerShell script to validate the package-directory name and all
  three hashes before extraction. Output examples now use a new versioned
  directory and never overwrite historical aligned-landmark CSVs.
- The first Windows debug then failed inside OpenFace because the release
  archive omitted the four binary CEN patch experts required by
  `main_ceclm_general.txt`. The script now fails before creating an output run
  when any dependency is missing, and records their size/SHA-256 after the
  official `download_models.ps1` step supplies them.

### FACE-S1 phase-1 full distribution and paired temporal review

- Added the read-only `face_usability.py` audit and completed
  `face_usability_phase1_v2` on all 300 videos and 493,141 WSL-local JPGs.
  It read no BDI labels, predictions, or AU values and generated no
  `face_usable` approval.
- Preserved 6,501 provenance-backed hard statuses: 2,365 pending raw-frame
  warp, 2,295 visible landmark failures, and 1,841 person-absent frames. The
  remaining 486,640 frames are explicitly `pending_threshold_review`.
- Fixed two debug-discovered pose defects before the full run: a 180-degree
  model/image coordinate convention mismatch and negative-depth iterative PnP
  solutions. The full run used 483,415 positive-depth ITERATIVE, 3,101 SQPnP,
  and 124 EPnP solutions; all output hashes were independently verified.
- Generated 11 train-only contact-sheet strata with 132 PENDING frames.
  Visual review supports pose/coverage/out-of-frame semantics but shows that
  blur is exposure/contrast-confounded and residual overlaps pose.
- Added `face_usability_temporal_review.py` and its CLI because a jump metric
  cannot be audited from one frame. The 12 exact `t-1/t` pairs recomputed the
  source metric within 1e-5 and showed mostly real motion/expression/blur, not
  systematic landmark teleport. Jump remains a review trigger, not an
  independent exclusion rule.
- Validation: focused tests `18 passed`; compileall and both CLI help checks
  passed. FACE-S2, threshold application, full warp/materialize, and training
  remain blocked pending train-only human labels and a hashed threshold
  manifest.
- Added a fail-closed dual-lane threshold-review preparation step. It
  deduplicated 132 contact rows into 124 train frames and separated global-face
  usability, landmark-geometry candidacy, and temporal-boundary decisions. The
  first local-crop wording was superseded before review because region polygons
  and margins are not frozen; v2 cannot approve any concrete crop. All 372
  decision cells remain PENDING; no threshold manifest was generated.

### Provenance-final frame audit and real raw-frame warp smoke

- Completed `frame_failure_recovery_safe_v1` on all 300 videos. It reproduced
  6,501 failed frames and 221 failure blocks, froze the q20/q80 safety band and
  q25/q75 targets, and recorded SHA-256 for the implementation and every output.
- Added `src/diagnostics/raw_frame_warp.py` and the fail-closed
  `audit_frame_recovery.py raw-warp-smoke` subcommand. Selection is train-only,
  requires exact source-presence coverage, and is limited to one to three frames.
- The smoke recovers raw-to-aligned transforms from valid landmark anchors,
  propagates landmarks through adjacent real raw-video frames with
  forward-backward LK checks, and warps pixels from the actual target frame.
  It never blends or copies aligned neighbor faces.
- Final v3 produced three deterministic `AUTO_PASS_REVIEW_REQUIRED` frames and
  rejected one candidate with an invalid previous raw anchor. Tracking validity
  was 0.98-1.00, target transform RMSE 0.98-1.25 px, and transform disagreement
  0.85-2.46 px. v1/v2/v3 derived JPEG hashes were identical.
- Contact-sheet inspection preserved subject identity, pose, glasses/headsets,
  microphone occlusion, and expression. The formal review field remains PENDING;
  no full repaired dataset, landmark rerun, materialize integration, or training
  authorization is implied. The next read-only task is FACE-S1.

### Route revision: face-valid clips and AU-semantic landmark local/global augmentation

- Replaced the proposed video-first random crop with deterministic mining of
  every qualified face-visible segment. The next audit must explicitly reject
  person absence, black/unreadable frames, major occlusion, extreme pose,
  severe out-of-frame faces, and unreliable landmarks.
- Clarified that more clips add training views but not independent subjects.
  Formal training must either normalize clip losses to unit weight per source
  video or aggregate clips with the same model before one video-level BDI loss.
- Removed AU intensity/presence values, AU sequences/features, auxiliary AU
  prediction, and AU supervision from the model data path. Retained AU/FACS as
  the semantic partition for four local RGB views: brow, eye-cheek,
  nose-upper-lip, and mouth-jaw. Validated aligned-space 68-point landmarks
  locate the complete regions frame by frame; neither AU values nor landmark
  coordinates are model inputs.
- Added `AVEC2014_SOURCE_DATA_QUALITY.md` to distinguish raw-video problems
  from aligned-placeholder and landmark/OpenFace derivation failures. It
  records presence, exposure, clipping, duration, versioning, and paper
  disclosure requirements.
- Existing AU-prefixed logs/scripts remain frame/coordinate/data-quality and
  region-tracking evidence. Current task names use `FACE-*` and `LM-*`, without
  removing the AU-semantic crop contract.

## 2026-07-16

### Validity-aware temporal slicing feasibility audit

- Added a read-only audit joining the frozen split, 300-video failure summary,
  6,501-frame failure manifest, and the fully reviewed 4,206-frame source-
  presence gate. No labels, predictions, images, or OpenFace processes are used.
- Confirmed that the historical `stride_head` contract reaches at most raw
  frame 1991 for `MAX_SEQ_LEN=2000,SAMPLE_STEP=10`; under the post-warp global
  capacity contract it leaves 83,840 valid frames (17.06%) unseen.
- Reject immediate slicing at current pure-black placeholders: it creates 527
  valid runs and 232 runs shorter than 600 frames. After approved raw-warp, the
  capacity estimate is 301 runs and 380 balanced non-overlap clips at a
  2000-frame ceiling; 66/300 videos require more than one clip.
- Froze the design direction as video-first stochastic training crop and
  deterministic video-level multi-clip aggregation. Training code remains
  blocked until provenance-final audit, raw-warp validation, and a rerun of the
  validity manifest.
- Validation: `pytest tests/test_validity_aware_slicing.py
  tests/test_temporal_sampling.py` -> 9 passed.

### AU-T0c log-family exposure recovery and fail-closed segment gate

- Replaced the formal gamma recovery route with two endpoint-preserving luma
  curves: underexposure uses `log(1+a*x)/log(1+a)` and overexposure uses
  `(exp(b*x)-1)/(exp(b)-1)`. The old gamma function and CSV columns remain only
  for historical reproducibility and are not called by formal materialization.
- Added monotone parameter fitting so one reviewed video/segment median maps to
  the frozen train-normal target. Per-frame fitting is prohibited.
- Rejected the first q10/q90 edge targets (`44.873/173.274`). The revised
  train-only protocol uses q20/q80 (`49.018/142.171`) as the post-transform
  acceptance band and places the fitted targets strictly inside it at q25/q75
  (`53.740/135.769`). Preview tables now report q90-q10 span retention and both
  clipping tails; stronger mapping is rejected when it only moves clipping to
  the opposite tail.
- Full-frame temporary review of all 22 candidates identified stable
  overexposure in `211_1_Freeform` and `242_1_Northwind`, but rejected a single
  whole-video inverse-log curve for `212_1_Freeform` (luma IQR `31.15`) and
  `223_1_Freeform` (`10.57`). Four underexposed videos also require priority
  segment review: `310_2`, `219_1`, `232_1`, and `250_1` Freeform.
- The audit now emits `tables/exposure_review_template.csv` with PENDING rows.
  Materialization fails closed unless every selected exposure candidate has
  complete, non-overlapping `REVIEWED` coverage. Allowed decisions are
  `stable_log`, `tone_only_overexposed`, and `keep_raw`.
- Added `scripts/preview_exposure_recovery.py` to write non-authorizing
  BEFORE/AFTER contact sheets from deterministic audit samples and sampled
  luma extrema. It also writes frame-level luma/clipping comparisons and never
  writes a formal derived dataset.
- Added optional all-frame preview evaluation. A real q25/q75 full scan over
  `225_2_Freeform` (450 visible frames) moved the frame-median center from
  `32.193` to `53.191`; `211_1_Freeform` (1,230 frames) moved from `224.308`
  to `136.316`. Both entered the q20/q80 safety band. For `211_1`, high-tail
  clipping fell from `0.1141` to `0.0442`, while low-tail clipping rose from
  `0.0152` to `0.0371`; remaining clipped highlights are irreversible and must
  stay visible in the report rather than be described as recovered detail.
- Added fail-closed relocated-root authorization for preview. The WSL copy at
  `/home/zhen/dataset/depression/avec/2014/face_images` is accepted only through
  the existing `EXACT_PASS` comparison: `493141/493141 EXACT_MATCH`, candidate
  manifest SHA-256 `07bc830c...59daabb`, PASS inventory, and an exact inventory
  root match. A real two-video relocated-root smoke passed and recorded all
  evidence hashes in `preview_manifest.json`.
- Every changed frame records curve type/parameter and reviewed segment bounds.
  Source images remain read-only, black padding is preserved, and overexposure
  is described only as tone normalization rather than clipped-detail recovery.
- Local validation: frame-recovery/source-presence tests `9 passed`; focused
  Python compilation and `git diff --check` passed. No formal repaired dataset
  or FaceLandmark rerun was produced.

### AU-T0c review of the six raw-detail-inconclusive videos

- Reviewed `218_1_Freeform`, `219_1_Northwind`, `219_3_Northwind`,
  `225_2_Northwind`, `310_2_Freeform`, and `328_1_Northwind` using full-frame
  luma series, extrema/boundary contact sheets, raw ROI/raw warp/aligned
  comparisons, and exact full-frame application of candidate fixed curves.
  No BDI label or validation/test metric was read.
- Approved whole-video `stable_log` for `218_1` (`a=2.053613`), `219_1`
  (`a=1.724186`), `225_2` (`a=1.712135`), and `328_1` (`a=2.135340`).
  Their visible-frame safety-band coverage was `94.91%`, `90.61%`, `98.11%`,
  and `100%`, respectively.
- Split `219_3_Northwind` into `1-65` (`a=4.810923`) and `66-1170`
  (`a=2.425732`); both segments reached `100%` visible-frame safety-band
  coverage under one fixed parameter per segment.
- Routed `310_2_Freeform` to `keep_raw`. Its luma IQR is `14.76` and the camera
  rises into and falls out of a bright plateau through gradual transitions;
  several different segmentations fit the series, so no unique coarse boundary
  is defensible without content-dependent preprocessing.
- Rejected extra segmentation for `219_1`: its darker intervals track continuous
  head-pose/shading changes. Rejected the apparent `225_2` dark segment at
  frames `110-137`: it is caused by a hand occluding the face. Brightening either
  case with a separate fitted curve would normalize behavior or occlusion rather
  than acquisition exposure.
- Wrote the partial reviewed manifest and full numeric report to
  `logs/au_region_tracking_audit/exposure_uncertain_six_review/`. At this
  checkpoint only the six automated-inconclusive routes were resolved; the
  immediately following entry records completion of the remaining 16 and the
  train-normal threshold. No source or derived image was written and OpenFace
  was not rerun.

### AU-T0c train-only temporal threshold and complete 22-video exposure gate

- Added `src/diagnostics/exposure_temporal_stability.py`,
  `scripts/audit_exposure_temporal_stability.py`, and focused tests. The command
  scans aligned JPGs read-only and does not read BDI labels, predictions,
  validation metrics, or test metrics.
- Full scan covered 92 train-normal reference videos plus all 22 exposure
  candidates. Train-normal luma-IQR quantiles were q50 `3.688544`, q75
  `6.000996`, q80 `6.717774`, q90 `10.243687`, and q95 `12.024606`.
  The preregistered whole-video eligibility threshold is q90; q95 is only an
  extreme-priority stratum and cannot relax q90.
- Candidate triage was 18 below q90, two q90-q95 (`223_1_Freeform`,
  `219_1_Freeform`), and two above q95 (`212_1_Freeform`, `310_2_Freeform`).
  Global IQR missed the short initial state in `219_3_Northwind`, confirming
  that representative/extreme/boundary review remains mandatory.
- Completed a 24-row reviewed manifest covering all 22 candidates. Seventeen
  underexposed videos use log curves; `219_1_Freeform` uses objective boundary
  `1494/1495` and `219_3_Northwind` uses `1-65/66-1170`. Stable overexposed
  `211_1_Freeform` and `242_1_Northwind` use tone-only inverse-log. `212_1`,
  `223_1`, and `310_2` stay raw.
- The `219_1_Freeform` boundary was selected by a label-free one-change
  least-squares scan with a 300-frame minimum segment; relative within-segment
  SSE reduction was `0.885674`. Content-driven changes such as head shading,
  face distance, and hand occlusion were not assigned separate parameters.
- Exact full-frame transform evaluation confirmed every approved non-identity
  segment median lies inside `[49.018280,142.171405]`. Visible-frame safety-band
  coverage ranges `75.29%-100%`; this is diagnostic and does not authorize
  per-frame adaptive normalization.
- Outputs are in `logs/au_region_tracking_audit/exposure_temporal_stability_q90_v1/`
  and `logs/au_region_tracking_audit/exposure_review_full_q90_v1/`. Related
  exposure/frame-recovery tests: `14 passed`. No source image was modified,
  no repaired dataset was materialized, and OpenFace was not rerun.

### AU-T0c raw-video versus aligned-JPG detail recoverability audit

- Added `src/diagnostics/raw_detail_recoverability.py`,
  `scripts/audit_raw_detail_recoverability.py`, and focused tests.
- Existing raw/aligned 68-point CSVs are used only for same-frame geometry via
  RANSAC similarity transforms; AU/pose/embedding values are not compared.
  OpenFace will be rerun with one frozen version after the input variant is
  finalized.
- Train-only calibration used 92 normal-exposure videos and 365 valid reference
  frames. Candidate evaluation covered 22 videos, 701 sampled frames, and 685
  valid comparisons; median transform inlier ratio/RMSE were `0.941/0.687 px`.
- A provisional q95-only pass marked 21 frames, but every apparent clipping
  advantage was below one percentage point. V2 therefore froze a `0.01`
  absolute practical-effect gate and retained the q95 train-normal thresholds.
- V2 result: 499 `raw_also_clipped`, 185 no raw recoverability evidence, one
  less-clipped but texture-inconclusive frame, 16 invalid comparisons, and zero
  `raw_detail_recoverable` frames. Video routing is 16
  `tone_only_or_keep_raw` plus six `inconclusive_keep_raw_until_review`; no
  `raw_space_tone_then_warp_candidate` exists.
- Formal outputs are in
  `logs/au_region_tracking_audit/raw_detail_recoverability_v2/`. Source data
  were not modified, OpenFace was not run, and output hash verification passed.

### Full original-video presence review and black-frame recovery gate correction

- Added `src/diagnostics/source_presence.py`, `scripts/audit_source_presence.py`,
  `scripts/summarize_source_presence_review.py`, and focused synthetic tests.
- Froze the original-video root as
  `D:\Project\dataset\AVEC2014\{train,dev,test}\{Freeform,Northwind}` and
  mapped it in WSL as `/mnt/d/Project/dataset/AVEC2014`.
- Full source contract passed for all 300 videos: every raw video is
  `640x480 @ 30 FPS`, and every raw frame count exactly matches the aligned-JPG
  count. No source video, aligned image, split, or label was modified.
- Split 4,206 pure-black aligned placeholders into 231 source-video runs over
  38 videos and generated one review contact sheet per run. Runs of length
  `>=10` outside `247_3` received an additional dense/all-frame review over
  1,605 decoded source frames; all 2,136 frames of the `247_3` mixed run were
  also reviewed in consecutive all-frame pages.
- Completed a non-overlapping, full-coverage human review manifest and expanded
  it to a 4,206-row frame gate:
  - 2,365 frames: `person_present_detection_failure -> raw_frame_warp_only`;
  - 1,841 frames: `person_absent -> keep_invalid`;
  - 0 mixed/ambiguous frames after segmentation.
- Only `247_3_Freeform` contains confirmed absence. Its long run is split as
  `2078-2143` person present, `2144-3984` empty room, and `3985-4213` person
  re-entering. The remaining 230 runs retain a visible person throughout.
- Corrected the previous aligned-neighbor recovery proposal: bidirectional
  aligned optical flow and previous-aligned-frame copy are now disabled for
  every pure-black frame because they can invent a face or erase real
  occlusion/pose/absence evidence. A future recovery smoke may only warp the
  actual raw frame using audited transform propagation.
- Formal outputs:
  `logs/au_region_tracking_audit/source_video_presence/` and
  `logs/au_region_tracking_audit/source_video_presence/full_review/`.
- Validation: source/recovery focused tests `8 passed`; Python compilation,
  report coverage/count checks, and `git diff --check` passed at this stage.

### AU-T0c initial frame-failure inventory and provisional recovery implementation

> Superseded recovery decision: the source-video review above disables all
> aligned-neighbor black-frame synthesis. The counts and exposure inventory in
> this historical entry remain valid; its optical-flow permission does not.

- Added `src/diagnostics/frame_recovery.py`, `scripts/audit_frame_recovery.py`,
  and focused synthetic tests.
- The `audit` command consumes existing aligned-image OpenFace CSV/JPG files
  and explicitly records `face_landmark_rerun_performed=false`; it does not
  launch OpenFace or modify source images.
- Formal full-data output was written to
  `logs/au_region_tracking_audit/frame_failure_recovery/`:
  - 300 videos, 6,501 failed frames, and 221 consecutive failure blocks;
  - 4,206 pure-black failures, 460 underexposed detection failures, 333
    overexposed detection failures, and 1,502 other visible failures;
  - 41 short pure-black blocks / 59 frames were initially proposed as
    optical-flow candidates; this permission was later revoked;
  - 18 videos are underexposed, 4 are overexposed, and 278 are normal.
- Exposure detection uses 32 deterministic samples per video.  Targets are
  frozen from train normal-video luma q10/q90 only: `44.873/173.274`.
- Materialization is non-destructive and supports a sparse changed-frame
  overlay or a complete hardlink/symlink/copy mirror.  Every repaired frame
  records raw/derived SHA-256 and all recovery parameters.
- Historical real-data materialization smoke (not authorized for formal data):
  - `207_2_Freeform` frames 52-53 were synthesized from frames 51/54 with
    median flow magnitude `2.4806 px` and median forward-backward error
    `0.4148 px`;
  - `225_2_Freeform` frame-median distribution center changed from `32.193`
    to `44.953` under one gamma;
  - `211_1_Freeform` changed from `224.308` to `173.926`.
- Only temporary `/tmp` derived samples were generated.  Full repaired data,
  a FaceLandmark rerun, AU-T1, and training remain blocked on manual review.
- Validation completed: focused pytest, Python compilation, real two-video
  audit smoke, short-black optical-flow smoke, and under/overexposure video
  materialization smoke.
- Provenance caveat: the first successful full run predates the final
  implementation/output SHA-256 manifest fields.  Three later full reruns
  (8/4/1 workers) were externally terminated by the desktop sandbox; the
  existing complete tables were not overwritten.  Re-run after commit in the
  stable local/server environment before paper-level freezing.

## 2026-07-14

### P0/P1/P2 video-level photometric normalization implementation

- Added an optional, deterministic video-level luminance normalization stage
  before existing input variants, resize/model normalization, and padding.
- Frozen matrix:
  - P0 `none`: original RGB control;
  - P1 `luma_center`: bounded video-level luminance-median alignment;
  - P2 `luma_center_contrast`: P1 plus bounded q10-q90 contrast alignment.
- The transform uses one mapping for all sampled frames in a video, excludes
  axis-connected near-black OpenFace padding from robust statistics, preserves
  those padding pixels, and changes RGB channels by the same luminance delta.
  The shared delta is gamut-limited per pixel so saturated colors preserve
  their channel differences instead of undergoing channel-wise clipping.
- The first matrix uses fixed canonical targets (`median=0.50`, `span=0.50`) and
  therefore does not fit any transform target from validation/test. P1/P2 do not
  claim to normalize color or skin tone; color constancy remains a later P3.
- Added standalone validation-only configs under
  `configs/photometric_normalization/`. P0/P1/P2 differ only in experiment name
  and photometric mode and use the same seed, split, backbone policy, optimizer,
  precision, severity-balanced loss, EarlyStopping, and checkpoint monitor.
- Local validation completed:
  - photometric/config tests: `48 passed`;
  - input/dataset/model/loss/training-policy regression selection: `120 passed`;
  - root `tests/` suite: `344 passed`; three failures and two errors remain in
    legacy end-to-end tests that import the absent `src.models.end_to_end`;
  - Python compilation and `git diff --check` passed;
  - a synthetic `[200, 3, 112, 112]` P2 tensor smoke preserved shape and uint8
    range.
- Real AVEC one-batch/training smoke remains server-only because the local
  `configs/local_paths.yaml` contains no dataset paths or backbone weights.
- Interpretation caveats are frozen: the strict padding mask is not a general
  connected-component segmentation, and the unchanged `mean=std=0.5` model
  normalization may differ from the external DeiT pretraining contract. The
  latter requires a separate factorial control and must not be changed inside
  P0/P1/P2.
- Reproducible server command (replace the final override for P1/P2):

```bash
python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml \
  --override configs/photometric_normalization/debug_smoke.yaml
```

- Before each real run, record `git rev-parse HEAD`,
  `git branch --show-current`, the exact command, GPU/device, and the generated
  `<LOG_DIR>/photometric_normalization*/<EXPERIMENT_NAME>/version_N/resolved_config.yaml`.

## 2026-06-12

### Debug smoke and config-path maintenance

- Completed config loader validation in the real project environment.
- Created/prepared `configs/local_paths.yaml` on the target machine from the
  ignored local paths workflow.
- Restored/fixed `src.datasets.dataset` imports required before smoke training.
- Audited import dependencies after restoring `src.datasets.dataset`.
- Verified that debug smoke training can run end to end in the server
  environment.
- Verified `scripts.diagnose` import checks:
  - `import scripts.diagnose`
  - `from scripts.diagnose import build_parser, load_config_from_args, run_diagnostic`
  - `python scripts/diagnose.py --help`

## 2026-06-13

### OpenFace behavior-representation research direction

- Reviewed recent experiment behavior from regression-only and backbone
  freeze/high-layer finetuning runs.
- Interpreted the generalization issue as likely shortcut learning inside
  OpenFace aligned face frames rather than only raw background overfitting.
- Decided to prioritize OpenFace quality diagnostics, input ablations,
  landmark/AU/pose/gaze baselines, and RGB + behavior late fusion before
  further backbone-layer search.
- Added `docs/RESEARCH_NOTES.md` to archive related papers and the next
  experiment roadmap.

### Shortcut Audit design

- Defined the Shortcut Audit Framework for validating non-depression shortcuts
  in OpenFace aligned face inputs.
- Scoped the first implementation to offline diagnostics: OpenFace quality
  summary, prediction-residual correlation, heatmaps, and markdown report.
- Added `docs/SHORTCUT_AUDIT_DESIGN.md` as the implementation blueprint.

### Shortcut Audit video-level alignment

- Found that short `subject_id` values such as `203_1` are ambiguous because
  OpenFace exports may contain both Freeform and Northwind files for the same
  subject/session.
- Updated the planned diagnostic alignment to prefer full `video_id` and only
  fall back to `subject_id` when the match is unique.

### Latest Shortcut Audit file review

- Reviewed the latest exported diagnostic files:
  - `test_predictions.csv`
  - `openface_quality_summary.csv`
  - `shortcut_audit_report.md`
  - `shortcut_merged.csv`
  - `shortcut_correlation.csv`
  - `shortcut_predictor_results.csv`
- Found that `test_predictions.csv` used IDs such as
  `203_2_Freeform_video_aligned`, while OpenFace summaries used IDs such as
  `203_2_Freeform_video`.
- The current Shortcut Audit matched 0 samples, so the generated shortcut risk
  level is invalid and must not be interpreted as evidence of low shortcut
  risk.
- A manual normalization check that removes the `_aligned` suffix matched
  100/100 prediction samples, confirming that the problem is an ID
  normalization gap rather than missing OpenFace files.
- Current prediction diagnostics still show strong range compression:
  MAE about 8.91, RMSE about 10.95, Pearson about 0.35, CCC about 0.29.
- Group-wise bias remains important:
  minimal samples are overpredicted on average, while severe samples are
  strongly underpredicted on average.
- OpenFace pose/gaze/AU/quality features show non-trivial diagnostic signal
  after manual matching; they should remain offline audit variables until a
  behavior-only baseline and leakage-safe grouped validation are established.

### Shortcut Audit `_aligned` video-id normalization fix

- Updated Shortcut Audit matching so prediction IDs such as
  `203_2_Freeform_video_aligned` normalize to `203_2_Freeform_video`.
- Added a focused test to ensure aligned prediction IDs match the correct
  Freeform/Northwind OpenFace summary row without falling back to ambiguous
  short `subject_id` matching.
- Verified with the latest local logs that 100/100 prediction rows match the
  OpenFace quality summary after normalization.

### Valid Shortcut Audit and representation analysis

- Re-ran Shortcut Audit after the `_aligned` video-id normalization fix.
- Verified that `shortcut_audit_report.md` now reports `Matched samples: 100`
  and `shortcut_merged.csv` contains 100 rows matched by full `video_id`.
- Current Shortcut Audit risk level is medium, with maximum absolute
  correlation about 0.418 between `AU07_c_mean` and `true_bdi`.
- Current RGB/MTL-Lite prediction remains range-compressed:
  MAE about 8.91, RMSE about 10.95, Pearson about 0.35, CCC about 0.29,
  prediction std about 6.13 versus true BDI std about 11.48.
- Group-wise bias:
  minimal samples are overpredicted on average by about +6.89, while severe
  samples are underpredicted on average by about -16.50.
- Most severe high-error cases include `246_1`, `359_1`, `237_1`, and
  `315_2`; these should be prioritized for case-study visualization.
- Freeform and Northwind aggregate metrics are similar, but same-subject task
  predictions can differ substantially. The largest observed task-pair
  difference is about 13.61, and high-difference cases include `237_1`,
  `247_1`, `247_3`, `224_1`, and `212_1`.
- In-sample shortcut-only linear/ridge predictors are very strong, but this is
  not a valid generalization estimate because the diagnostic currently has 100
  samples and 94 numeric OpenFace features.
- A manual grouped-CV check suggests shortcut-only ridge remains close to, but
  slightly weaker than, the current RGB model. This supports treating OpenFace
  behavior/quality features as a medium shortcut risk and motivates a formal
  grouped-CV diagnostic plus behavior-only baseline.

### Grouped-CV shortcut-only predictor implementation

- Added subject-level grouped CV design to `docs/SHORTCUT_AUDIT_DESIGN.md`.
- Implemented grouped-CV shortcut-only predictor diagnostics in Shortcut Audit.
- The grouped CV logic keeps all videos from the same `subject_id` in the same
  fold to reduce Freeform/Northwind leakage.
- Shortcut Audit now writes both the combined `shortcut_predictor_results.csv`
  and a focused `shortcut_predictor_grouped_cv.csv`.
- The markdown report now separates in-sample predictor diagnostics from
  grouped-CV predictor diagnostics.
- Local validation:
  - `python -m compileall src scripts tests` passed.
  - Direct grouped-CV smoke passed.
  - Temporary end-to-end Shortcut Audit smoke generated the grouped-CV CSV and
    report section successfully.
- Local bundled Python still lacks `pytest`, so full pytest validation should
  be run in the server environment.

### P0 剩余任务设计归档

- 根据最新 grouped-CV shortcut-only predictor 结果，当前 shortcut 风险维持为
  medium：OpenFace 统计特征与标签、预测和误差存在中等相关，但不能单独接近
  RGB/MTL-Lite 测试表现。
- 后续 P0 不再只围绕 in-sample shortcut predictor 或 backbone 解冻层数展开，
  而是优先定位预测范围压缩、severe 系统性低估、minimal 系统性高估和
  Freeform/Northwind 同一 subject 预测不一致。
- 已将 P0-2 case study manifest、P0-3 input ablation protocol、
  P0-4 behavior-only baseline interface 的目标、输出、字段和判读原则写入
  `docs/TODO.md`、`docs/SHORTCUT_AUDIT_DESIGN.md`、`docs/CURRENT_STATUS.md`
  和 `docs/CODEX_CONTEXT.md`。
- 本次仅进行文档设计，不修改训练代码、不修改训练超参数、不修改
  `configs/local_paths.yaml`，也不删除或覆盖任何实验结果。

### P0-2 case study manifest implementation

- Added offline case-study manifest generation for high-error and
  task-inconsistency analysis.
- New module: `src/diagnostics/case_studies.py`.
- Regression diagnostics now emit `case_study_manifest.csv` and
  `case_study_manifest.md` next to the regression plots.
- Shortcut Audit now emits `tables/case_study_manifest.csv` and
  `reports/case_study_manifest.md`.
- Manifest case types:
  - `severe_underestimate`
  - `minimal_overestimate`
  - `task_inconsistency`
  - `low_error_reference`
- Added tests for manifest selection and output wiring.
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Direct manifest smoke passed.
  - Direct Shortcut Audit smoke generated `case_study_manifest.csv` and
    `case_study_manifest.md`.
  - Import check for `case_studies`, `regression`, and `shortcut_audit` passed.
  - Local bundled Python still lacks `pytest`; run the focused pytest command
    on the server environment.

### P0-3 input ablation variant implementation

- Added optional RGB input ablation support through `DATASET.INPUT_VARIANT`.
- New module: `src/datasets/input_variants.py`.
- `AVECDataset` now applies the selected variant before resize/normalize and
  before training augmentations.
- Default behavior remains `rgb`, so existing configs keep the same data path.
- Supported variants:
  - `rgb`
  - `grayscale`
  - `blur`
  - `center_mask`
  - `boundary_erased`
- `landmark_heatmap` is intentionally reserved for the OpenFace
  landmark/behavior baseline route; configuring it in the RGB dataset raises a
  clear error instead of silently generating a fake landmark input.
- Added `DATASET.INPUT_VARIANT: "rgb"` to `configs/avec2014_base.yaml`.
- Added `tests/test_input_variants.py`.
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Local bundled Python lacks `torch`, `pytorch_lightning`, and `pytest`; run
    input-variant pytest and dataset import checks on the server environment.

### P0-4 behavior-only baseline interface implementation

- Added independent OpenFace behavior-only baseline route.
- New dataset module: `src/datasets/openface_features.py`.
  - Matches OpenFace CSV files to split video IDs.
  - Builds AU/pose/gaze/landmark/quality temporal features.
  - Appends temporal delta by default and optional acceleration.
  - Computes normalization statistics from the training split only.
- New model module: `src/models/behavior_baseline.py`.
  - Feature projection + GRU temporal encoder + mask-aware pooling.
  - BDI regression head and optional ordinal auxiliary head.
- New runner and entry:
  - `src/trainers/behavior_baseline_runner.py`
  - `scripts/train_behavior_baseline.py`
- New config:
  - `configs/behavior_baseline.yaml`
- New tests:
  - `tests/test_openface_features.py`
  - `tests/test_behavior_baseline.py`
- Validation:
  - `python -m compileall src scripts tests` passed with the bundled Codex
    Python runtime.
  - Local bundled Python lacks `torch`, `pytorch_lightning`, and `pytest`; run
    focused behavior baseline tests and import checks on the server environment.

## 2026-06-14

### Behavior-only baseline result review

- Reviewed the latest `behavior_metrics.csv` exported from the OpenFace
  behavior-only baseline run.
- Test metrics:
  - MAE about 9.93.
  - RMSE about 12.86.
  - CCC about 0.151.
- Best validation RMSE occurred around epoch 65:
  - val MAE about 9.94.
  - val RMSE about 12.38.
  - val CCC about 0.324.
  - corresponding train MAE about 2.17, train RMSE about 2.74, train CCC about
    0.975.
- Interpretation:
  - The behavior-only baseline currently overfits strongly.
  - Complete OpenFace feature sets should not be treated as clean behavioral
    representations.
  - Raw landmark coordinates and static facial geometry may carry identity or
    subject-specific shortcuts.
  - Late fusion should wait until feature-group ablation identifies a stable
    behavior subset.
- Updated documentation:
  - `docs/CURRENT_STATUS.md`
  - `docs/TODO.md`
  - `docs/CODEX_CONTEXT.md`
  - `docs/RESEARCH_NOTES.md`
  - `docs/SHORTCUT_AUDIT_DESIGN.md`
  - `docs/BUG_LOG.md`

### P0 behavior prediction export implementation

- Added behavior baseline val/test prediction export after best-checkpoint test
  evaluation.
- Prediction CSV files are written under:
  `<LOG_DIR>/default/behavior_baseline/version_0/diagnostics/behavior/`.
- Exported files:
  - `val_predictions.csv`
  - `test_predictions.csv`
- Prediction rows now include `video_id`, `subject_id`, `task_name`,
  `true_bdi`, `pred_bdi`, `residual`, `abs_error`, and `severity_group`.
- Extended the shared prediction-table writer with an optional `task_name`
  field while preserving existing callers that do not provide task names.
- This change does not alter model forward, losses, metrics, training
  hyperparameters, checkpoint selection, or `configs/local_paths.yaml`.

### P0 behavior feature-set ablation interface

- Added `BEHAVIOR_FEATURES.FEATURE_SET` with default value `custom`.
- Supported named feature sets:
  - `quality_only`
  - `au_only`
  - `pose_gaze_only`
  - `raw_landmark_only`
  - `landmark_delta_only`
  - `au_landmark_delta`
  - `all_without_raw_landmarks`
- The named feature sets make future ablations runnable with a minimal override
  instead of maintaining many near-duplicate YAML files.
- `landmark_delta_only` excludes raw landmark coordinates, and
  `all_without_raw_landmarks` keeps non-landmark raw features while retaining
  temporal deltas for all selected features.
- Default `custom` behavior preserves the existing boolean feature flags and
  therefore does not change previous behavior baseline runs.

### P0 RGB-vs-behavior prediction comparison interface

- Added offline comparison module:
  - `src/diagnostics/behavior_comparison.py`
  - `scripts/compare_behavior_predictions.py`
- The comparison aligns RGB/MTL-Lite and behavior-only predictions by
  normalized `video_id`, including compatibility with `_aligned` processing
  suffixes.
- Outputs:
  - `rgb_behavior_prediction_comparison.csv`
  - `rgb_behavior_prediction_summary.csv`
- The summary reports RGB and behavior MAE/RMSE/Pearson/CCC, severity-group
  metrics, and counts where RGB, behavior, or neither is better.

## 2026-06-15

### RGB input ablation result review

- Reviewed the first RGB input ablation batch:
  - `rgb`
  - `grayscale`
  - `blur`
  - `center_mask`
  - `boundary_erased`
- All reviewed runs used comparable core settings: same split, seed, MTL-Lite
  regression-only route, DeiT-tiny backbone weights, frozen backbone with the
  last 2 transformer blocks trainable, max sequence length 2000, and the same
  checkpoint-based evaluation style.
- Key test results:
  - `center_mask`: MAE about `7.94`, RMSE about `10.16`, Pearson about `0.51`,
    CCC about `0.48`.
  - `rgb`: MAE about `8.91`, RMSE about `10.95`, Pearson about `0.35`, CCC
    about `0.29`.
  - `boundary_erased`: close to or slightly better than `rgb`, but weaker than
    `center_mask`.
  - `blur` and `grayscale`: worse than `rgb`.
- Interpretation:
  - The improvement from `center_mask` suggests that peripheral/crop/alignment
    artifacts are hurting generalization.
  - The degradation from `grayscale` and `blur` suggests that simple color
    shortcut or fine texture shortcut is not the only explanation.
  - Severe underestimation remains unresolved, so input artifact mitigation is
    only one part of the failure analysis.

### OpenFace black-padding artifact hypothesis

- Reviewed an OpenFace aligned face sample showing pure black fill around the
  face contour and black microphone occlusion inside the crop.
- Updated the research hypothesis from generic "background shortcut" to a more
  specific OpenFace artifact mechanism:
  - black padding around aligned face contours;
  - black occluder regions such as microphones;
  - hard pixel discontinuities at crop and mask boundaries;
  - possible ViT/DeiT sensitivity to these structured high-contrast edges.
- Current priority is to explain RGB model overfitting before adding RGB +
  behavior late fusion or additional auxiliary tasks.

### Black artifact ablation implementation

- Extended `src/datasets/input_variants.py` with black artifact variants:
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
- Added aliases:
  - `black_fill_gray`
  - `black_fill_mean`
  - `black_fill_blur`
  - `soft_mask`
  - `inner_crop`
- Added input ablation configs:
  - `configs/input_ablation/black_to_gray.yaml`
  - `configs/input_ablation/black_to_mean.yaml`
  - `configs/input_ablation/black_to_blur.yaml`
  - `configs/input_ablation/soft_center_mask.yaml`
  - `configs/input_ablation/inner_crop_resize.yaml`
- Added tests for black replacement, soft masks, inner crop behavior, and
  aliases in `tests/test_input_variants.py`.

### Black artifact diagnostic implementation

- Added offline audit module:
  - `src/diagnostics/black_artifacts.py`
  - `scripts/audit_black_artifacts.py`
- The audit samples aligned frames and writes:
  - `tables/black_artifact_summary.csv`
  - `tables/black_artifact_merged.csv`
  - `tables/black_artifact_correlation.csv`
  - `reports/black_artifact_audit_report.md`
- Per-video features include:
  - mean/std black pixel ratio;
  - border black pixel ratio;
  - center black pixel ratio;
  - black-boundary edge ratio;
  - frame-to-frame black ratio delta.
- The merged audit correlates these artifact statistics with `true_bdi`,
  `pred_bdi`, `residual`, and `abs_error`.

### Validation

- Local compile validation passed for:
  - `src/datasets/input_variants.py`
  - `src/diagnostics/black_artifacts.py`
  - `scripts/audit_black_artifacts.py`
  - `tests/test_input_variants.py`
- Local bundled Python still lacks `torch` and `pytest`; focused pytest should
  be run on the server:

```bash
python -m pytest tests/test_input_variants.py
```

### Black artifact ablation result review

- Reviewed second-round RGB artifact ablations:
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
- Compared them against previous `rgb`, `center_mask`, and
  `boundary_erased` runs under the same core training settings.
- Main results:
  - `center_mask` remains best overall: MAE about `7.94`, RMSE about `10.16`,
    Pearson about `0.51`, CCC about `0.48`.
  - `black_to_gray` is the best new black-artifact variant: MAE about `8.34`,
    RMSE about `10.62`, CCC about `0.39`.
  - `soft_center_mask` improves severe bias more than most variants but
    overpredicts minimal samples, leading to worse overall MAE than
    `center_mask`.
  - `black_to_mean` and `inner_crop_resize` degrade test performance and should
    not be prioritized as the next main direction.
- Interpretation:
  - Replacing black pixels helps compared with raw `rgb`, but does not explain
    the full `center_mask` gain.
  - The artifact risk is likely a mixture of boundary fill, crop shape,
    peripheral non-behavior regions, face contour, pose/scale remnants, and
    subject-specific appearance.

### Black artifact audit result review

- Reviewed `black_artifact_audit_report.md`,
  `black_artifact_summary.csv`, `black_artifact_merged.csv`, and
  `black_artifact_correlation.csv`.
- The audit matched all expected test videos:
  - Videos summarized: `100`.
  - Missing videos: `0`.
  - Matched prediction rows: `100`.
- Black regions are common in aligned frames:
  - `black_ratio_mean` average about `0.24`.
  - `black_border_ratio_mean` average about `0.44`.
  - `black_center_ratio_mean` average about `0.02`.
- Maximum absolute correlation is weak, about `0.207`, so black artifacts are
  not a strong single-variable explanation for RGB overfitting.
- However, high-border-black quartile samples show larger average error than
  low-border-black quartile samples, about `12.29` versus `7.45`.
- Important correction:
  - `black_border_ratio_mean` is the cleaner OpenFace artifact indicator.
  - `black_center_ratio_mean` is semantically mixed because center black pixels
    may represent nostrils, mouth shadows, beard, natural facial shadows,
    microphones, or true occlusions.
- Next experiment plan:
  - Implement `border_black_to_gray`.
  - Implement `border_black_feather`.
  - Implement `center_mask_black_to_gray`.
  - Keep center black pixels unchanged unless a case study confirms they are
    preprocessing artifacts rather than facial structure or real occlusion.

### Border-connected black artifact ablation implementation

- Added three border-connected black artifact input variants:
  - `border_black_to_gray`
  - `border_black_feather`
  - `center_mask_black_to_gray`
- Implementation location:
  - `src/datasets/input_variants.py`
- New configs:
  - `configs/input_ablation/border_black_to_gray.yaml`
  - `configs/input_ablation/border_black_feather.yaml`
  - `configs/input_ablation/center_mask_black_to_gray.yaml`
- Added focused tests in `tests/test_input_variants.py`:
  - border-connected black pixels are replaced;
  - center black pixels remain unchanged;
  - feathering softens boundary-adjacent pixels;
  - `center_mask_black_to_gray` uses a gray outside region rather than a hard
    black outside region.
- Local validation:
  - `python -m compileall src/datasets/input_variants.py tests/test_input_variants.py` passed with the bundled Codex Python runtime.
  - Local Python environment still lacks `pytest` and `torch`; run the focused
    pytest command on the server environment.

### RGB overfitting factor map documentation

- Consolidated the broader RGB overfitting hypothesis beyond black artifacts.
- Added a multi-factor research map to `docs/RESEARCH_NOTES.md`, covering:
  - identity and static appearance shortcuts;
  - OpenFace alignment geometry and crop artifacts;
  - pose, gaze, confidence, success, and tracking quality;
  - video length, temporal sampling, and padding;
  - Freeform/Northwind task-context differences;
  - prediction compression and severity calibration;
  - ViT/DeiT patch-level shortcut sensitivity;
  - targeted robustness augmentation.
- Expanded `docs/SHORTCUT_AUDIT_DESIGN.md` with audit modules for:
  - identity/static appearance;
  - alignment geometry;
  - pose/gaze/tracking quality;
  - temporal sampling;
  - task consistency;
  - severity calibration.
- Added follow-up task queues to `docs/TODO.md`.
- Updated `docs/CURRENT_STATUS.md` and `docs/CODEX_CONTEXT.md` so future work
  treats black borders as one risk factor rather than a complete explanation.

### P0 temporal sampling audit implementation

- Implemented the first P0 multi-factor audit for video length, temporal
  sampling, truncation, and padding.
- New module:
  - `src/diagnostics/temporal_sampling.py`
- New CLI:
  - `scripts/audit_temporal_sampling.py`
- New tests:
  - `tests/test_temporal_sampling_audit.py`
- The audit follows the actual `AVECDataset` selection rule:
  - `sampled_frame_count = len(frames[::SAMPLE_STEP])`
  - `model_max_len = MAX_SEQ_LEN // SAMPLE_STEP`
  - `selected_frame_count = min(sampled_frame_count, model_max_len)`
  - padding and truncation are computed from the model-visible temporal length.
- Outputs:
  - `tables/temporal_sampling_summary.csv`
  - `tables/temporal_sampling_merged.csv`
  - `tables/temporal_sampling_correlation.csv`
  - `tables/temporal_sampling_group_summary.csv`
  - `reports/temporal_sampling_audit_report.md`
- Local validation:
  - `python -m compileall src/diagnostics/temporal_sampling.py scripts/audit_temporal_sampling.py tests/test_temporal_sampling_audit.py` passed with the bundled Codex Python runtime.
  - Direct smoke test generated all expected outputs.
  - Local Python still lacks `pytest`; run focused pytest on the server.

### P0 temporal sampling ablation implementation

- Added configurable temporal sampling strategies while preserving the
  historical default behavior.
- New module:
  - `src/datasets/temporal_sampling.py`
- Updated dataset:
  - `src/datasets/dataset.py`
- New config key:
  - `PROCESS_TEMPORAL.SAMPLING_STRATEGY`
- Supported strategies:
  - `stride_head`: historical `frames[::SAMPLE_STEP][:MAX_SEQ_LEN // SAMPLE_STEP]`;
  - `uniform`: uniformly sample model-visible frames across the whole video;
  - `first`: contiguous first crop;
  - `middle`: contiguous middle crop;
  - `random`: deterministic random contiguous crop keyed by video path.
- New training overrides:
  - `configs/temporal_sampling/uniform_256.yaml`
  - `configs/temporal_sampling/uniform_512.yaml`
  - `configs/temporal_sampling/uniform_1024.yaml`
  - `configs/temporal_sampling/first_crop.yaml`
  - `configs/temporal_sampling/middle_crop.yaml`
  - `configs/temporal_sampling/random_crop.yaml`
- New tests:
  - `tests/test_temporal_sampling.py`
  - extended `tests/test_temporal_sampling_audit.py`
- Local validation:
  - Compile validation passed.
  - Direct strategy smoke passed.
  - Direct uniform temporal audit smoke passed.
  - Server still needs focused pytest and actual training ablations.

### Border-connected black ablation result review

- Reviewed:
  - `rgb_ablation_border_black_to_gray`
  - `rgb_ablation_border_black_feather`
  - `rgb_ablation_center_mask_black_to_gray`
- All three runs are comparable to previous RGB ablations in core settings.
- Main results:
  - `center_mask_black_to_gray`: MAE about `7.73`, RMSE about `10.02`,
    Pearson about `0.52`, CCC about `0.45`.
  - `center_mask`: MAE about `7.94`, RMSE about `10.16`, Pearson about
    `0.51`, CCC about `0.48`.
  - `border_black_feather`: MAE about `8.04`, RMSE about `10.13`, Pearson
    about `0.50`, CCC about `0.41`.
  - `border_black_to_gray`: weaker than feathering, MAE about `8.69`.
- Interpretation:
  - `center_mask_black_to_gray` is currently best by MAE/RMSE/Pearson, but it
    mainly improves minimal/mild samples and worsens severe underestimation.
  - `center_mask` remains more balanced and retains the best CCC.
  - `border_black_feather` supports the boundary-softening hypothesis better
    than hard gray replacement.
  - Input artifact mitigation and severity calibration must be treated as
    separate problems.

### P0 prediction run summary implementation

- Added reusable multi-run prediction summary diagnostics:
  - `src/diagnostics/prediction_runs.py`
  - `scripts/summarize_prediction_runs.py`
  - `tests/test_prediction_runs.py`
- Outputs:
  - `tables/prediction_run_summary.csv`
  - `tables/severity_bias_summary.csv`
  - `tables/task_consistency_summary.csv`
  - `tables/pairwise_baseline_improvement.csv`
  - `reports/prediction_runs_report.md`
- The tool records true/pred mean/std, overall MAE/RMSE/Pearson/CCC,
  severity bias, Freeform/Northwind task consistency, and pairwise improvement
  against an optional baseline run.

### P0 RGB input ablation unified summary

- Ran `scripts/summarize_prediction_runs.py` on all currently available RGB
  input ablation prediction CSV files.
- Included runs:
  - `rgb`
  - `gray_scale`
  - `blur`
  - `boundary_erased`
  - `center_mask`
  - `black_to_gray`
  - `black_to_mean`
  - `black_to_blur`
  - `soft_center_mask`
  - `inner_crop_resize`
  - `border_black_feather`
  - `border_black_to_gray`
  - `center_mask_black_to_gray`
- Generated unified tables:
  - `prediction_run_summary.csv`
  - `severity_bias_summary.csv`
  - `task_consistency_summary.csv`
  - `pairwise_baseline_improvement.csv`
  - `prediction_runs_report.md`
- Current ranking by test MAE:
  `center_mask_black_to_gray` < `center_mask` < `border_black_feather` <
  `black_to_gray` < `soft_center_mask` < `border_black_to_gray` < `rgb`.
- Interpretation remains conservative:
  - `center_mask_black_to_gray` is best overall by MAE/RMSE/Pearson, but
    worsens severe underestimation.
  - `center_mask` remains the most balanced candidate by CCC and task
    consistency.
  - `gray_scale`, `blur`, `inner_crop_resize`, and `black_to_mean` do not
    support a single-factor explanation based only on color, texture, or
    peripheral cropping.
  - Severity calibration remains a separate P0/P2 issue from input artifact
    mitigation.

### High-priority overfitting audit review

- Reviewed the remaining overfitting-related ablation and validation queue.
- Decision: do not keep prioritizing additional RGB mask variants. The black
  artifact line has enough evidence to support a bounded sub-conclusion, but
  it cannot explain prediction compression or severe underestimation alone.
- Promoted the following items to the next high-priority queue:
  - split / subject integrity audit;
  - temporal sampling audit and fixed-frame / temporal-crop ablations;
  - training overfit curve summary across runs;
  - OpenFace alignment geometry audit;
  - embedding identity paired-task retrieval;
  - severity calibration verification;
  - task inconsistency mixed-factor audit.
- Updated `docs/TODO.md`, `docs/CURRENT_STATUS.md`,
  `docs/RESEARCH_NOTES.md`, `docs/SHORTCUT_AUDIT_DESIGN.md`, and
  `docs/CODEX_CONTEXT.md` with purpose, outputs, interpretation rules, and
  recommended execution order.
- Key interpretation:
  RGB overfitting should now be treated as a multi-factor mechanism involving
  OpenFace alignment geometry, identity/static appearance, temporal sampling,
  task context, and label-distribution compression. Border artifacts remain an
  important visible entry point, not a sufficient explanation.

### RGB overfitting audit plan consolidation

- Added canonical plan document:
  - `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
- Reorganized related documentation roles:
  - `RGB_OVERFITTING_AUDIT_PLAN.md`: authoritative research plan and
    experiment ordering;
  - `CURRENT_STATUS.md`: current status summary;
  - `TODO.md`: executable task checklist;
  - `SHORTCUT_AUDIT_DESIGN.md`: audit input/output/field specification;
  - `RESEARCH_NOTES.md`: paper background and research rationale;
  - `CODEX_CONTEXT.md`: handoff context for future Codex sessions.
- Clarified that black border / black padding is one possible overfitting
  factor, not the main thesis.
- Updated the current execution order to:
  split integrity -> temporal sampling -> training overfit curves ->
  alignment geometry -> embedding identity retrieval -> severity calibration ->
  task inconsistency mixed-factor audit.

### P0-A split integrity audit implementation

- Implemented offline split / subject integrity audit:
  - `src/diagnostics/split_integrity.py`
  - `scripts/audit_split_integrity.py`
  - `tests/test_split_integrity.py`
- The audit checks subject overlap across train/val/test, duplicate video ids,
  missing or invalid BDI label files, optional aligned image directory
  existence, and optional prediction-to-split alignment.
- Expected outputs:
  - `tables/split_video_manifest.csv`
  - `tables/split_subject_overlap.csv`
  - `tables/split_label_distribution.csv`
  - `tables/split_prediction_alignment.csv`
  - `reports/split_integrity_report.md`
- Interpretation rule:
  `split_integrity_report.md` should report `status: PASS` before later RGB
  overfitting diagnostics are interpreted as valid subject-disjoint
  generalization evidence.
- Local validation:
  compileall passed for `src/diagnostics/split_integrity.py`,
  `scripts/audit_split_integrity.py`, and `tests/test_split_integrity.py`;
  direct smoke passed on synthetic split/label/image/prediction data. Local
  bundled Python does not include `pytest`, so formal pytest remains
  server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_split_integrity.py

python scripts/audit_split_integrity.py \
  --split-file /path/to/dataset_split.json \
  --label-dir /path/to/labels \
  --image-root /path/to/aligned/frame/root \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/split_integrity
```

### P0-B training overfit summary implementation

- Implemented offline training overfit summary:
  - `src/diagnostics/training_overfit.py`
  - `scripts/summarize_training_overfit.py`
  - `tests/test_training_overfit.py`
- The audit reads Lightning-style `metrics.csv`, aggregates sparse metric rows
  by epoch, selects the best validation epoch by `val_RMSE_epoch` when
  available, then falls back to `val_MAE_epoch` and `val_loss`.
- Expected outputs:
  - `tables/training_overfit_summary.csv`
  - `tables/training_curve_gap_by_run.csv`
  - `reports/training_overfit_report.md`
- Interpretation rule:
  if a run improves test MAE but has a larger train/val gap or
  `overfit_after_best_val=True`, treat the result as possible prediction bias
  or checkpoint behavior rather than direct evidence of better generalization.
- Local validation:
  compileall passed for `src/diagnostics/training_overfit.py`,
  `scripts/summarize_training_overfit.py`, and `tests/test_training_overfit.py`;
  direct smoke passed on synthetic Lightning-style metrics rows. Local bundled
  Python does not include `pytest`, so formal pytest remains server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_training_overfit.py

python scripts/summarize_training_overfit.py \
  --output-dir analysis_outputs/training_overfit_summary \
  --run rgb=/path/to/rgb/metrics.csv \
  --run center_mask=/path/to/center_mask/metrics.csv \
  --run center_mask_black_to_gray=/path/to/center_mask_black_to_gray/metrics.csv \
  --run border_black_feather=/path/to/border_black_feather/metrics.csv \
  --run behavior=/path/to/behavior_baseline/metrics.csv
```

### P0-C alignment geometry audit implementation

- Implemented offline OpenFace alignment geometry audit:
  - `src/diagnostics/alignment_geometry.py`
  - `scripts/audit_alignment_geometry.py`
  - `tests/test_alignment_geometry.py`
- The audit reads OpenFace landmark columns `x_*` and `y_*`, computes landmark
  bbox width/height/area/aspect, face center offset, eye distance,
  normalized face scale, confidence/success summary, and landmark jitter.
- Expected outputs:
  - `tables/alignment_geometry_summary.csv`
  - `tables/alignment_geometry_merged.csv`
  - `tables/alignment_geometry_correlation.csv`
  - `tables/alignment_geometry_group_summary.csv`
  - `reports/alignment_geometry_audit_report.md`
- Interpretation rule:
  if geometry variables correlate with `residual` or `abs_error`, treat
  OpenFace alignment geometry as a shortcut or mixed-factor risk rather than
  attributing RGB overfitting only to black padding artifacts.
- Local validation:
  compileall passed for `src/diagnostics/alignment_geometry.py`,
  `scripts/audit_alignment_geometry.py`, and `tests/test_alignment_geometry.py`;
  direct smoke passed on synthetic OpenFace CSV and prediction rows. Local
  bundled Python does not include `pytest`, so formal pytest remains
  server-side.
- Next server-side commands:

```bash
python -m pytest tests/test_alignment_geometry.py

python scripts/audit_alignment_geometry.py \
  --predictions <LOG_DIR>/default/rgb/version_0/diagnostics/regression/test_predictions.csv \
  --openface-root /path/to/openface_csv_root \
  --output-dir <LOG_DIR>/default/rgb/version_0/diagnostics/alignment_geometry \
  --frame-width 112 \
  --frame-height 112 \
  --sample-step 1
```

### P0-A/B/C server results and geometry scale correction

Split integrity audit:

- `status: PASS`
- `300` videos, `150` subjects, `3` splits.
- `overlapping_subjects = 0`, `duplicate_video_rows = 0`, `missing_labels = 0`,
  `invalid_labels = 0`, `missing_image_dirs = 0`.
- Prediction alignment matched `100/100` test prediction rows.
- Interpretation: current RGB overfitting analysis is not explained by split
  leakage, label missingness, duplicate video ids, or prediction alignment
  mismatch.

Training overfit summary:

- RGB-family runs all reported `overfit_after_best_val=True`.
- `rgb` best validation RMSE occurred early at epoch `6`; after that, train
  RMSE kept improving while validation RMSE degraded.
- `center_mask_black_to_gray` had the best validation RMSE among the compared
  runs but still overfit after best validation.
- Behavior baseline showed a very large train/val gap, supporting the concern
  that full OpenFace behavior features contain strong subject/static geometry
  memorization signals.

Alignment geometry audit:

- `300` OpenFace videos summarized.
- `100/100` test prediction rows matched.
- Max absolute correlation: about `0.3746`.
- Top findings:
  - `landmark_bbox_height_mean` vs `true_bdi`: `r = 0.3746`
  - `landmark_bbox_area_mean` vs `true_bdi`: `r = 0.3410`
  - `landmark_bbox_width_mean` vs `true_bdi`: `r = 0.3041`
  - `eye_distance_mean` vs `true_bdi`: `r = 0.2991`
  - `landmark_bbox_height_mean` vs `residual`: `r = -0.2605`

Coordinate scale correction:

- Current OpenFace CSV landmark coordinates are not `112 x 112` aligned-frame
  coordinates.
- Observed examples: `x` range approximately `150-643`, `y` range
  approximately `-11-582`, while aligned jpg inputs are `112 x 112`.
- OpenFace run metadata reported camera parameters `500,500,320,240`, which
  indicates a likely source coordinate frame of approximately `640 x 480`.
- The alignment geometry audit was rerun with `--frame-width 640` and
  `--frame-height 480`.
- Therefore, the current geometry result should be interpreted as
  `pre-alignment detection geometry confound`, not as direct 112x112 input
  landmark geometry.
- Directly interpretable metrics: bbox width/height/area/aspect, eye distance,
  and landmark jitter in the OpenFace detection coordinate system.
- After the `640 x 480` rerun, `normalized_face_scale_mean` and
  `face_center_offset_*` are interpretable as relative pre-alignment detection
  geometry metrics. They still must not be described as 112x112 input-space
  landmark coordinates.
- Severity group means after the `640 x 480` rerun:
  - minimal: `scale=0.319`, `residual=+6.89`
  - mild: `scale=0.337`, `residual=-1.86`
  - moderate: `scale=0.399`, `residual=-7.66`
  - severe: `scale=0.364`, `residual=-16.50`
- Recommended follow-up implementation: add raw coordinate ranges and relative
  geometry ratios such as `eye_distance_to_bbox_height_ratio`.

### P0 temporal sampling audit server result

- Reviewed the RGB temporal sampling audit output generated on the current
  test prediction CSV.
- Matching quality was valid:
  - Videos summarized: `100`.
  - Matched prediction rows: `100`.
  - Missing videos: `0`.
  - Maximum absolute correlation: about `0.2197`.
- Main correlations:
  - `truncated_frame_count` vs `pred_bdi`: about `r = -0.2197`.
  - `truncated_ratio` vs `pred_bdi`: about `r = -0.2090`.
  - `raw_to_selected_ratio` vs `pred_bdi`: about `r = 0.2090`.
  - `sampled_frame_count` / `frame_count` vs `pred_bdi`: about
    `r = -0.2073`.
- Interpretation:
  - Temporal length and truncation are weak-to-moderate confounds.
  - Longest videos tend to receive lower predictions and higher errors, even
    though they have no padding; this points toward truncation, head-biased
    sampling, or diluted key behavior rather than padding alone.
  - Freeform videos are longer and less padded, while Northwind videos are
    shorter and more padded, but task-level absolute errors are similar.
  - Temporal sampling should remain a P0 ablation target, but it should not be
    treated as the sole cause of RGB overfitting or severe underestimation.
- Next:
  - Run `uniform_256`, `uniform_512`, `uniform_1024`, `first_crop`,
    `middle_crop`, and `random_crop` training ablations.
  - Summarize each run with `scripts/summarize_prediction_runs.py`.
### Local occlusion and static-appearance shortcut direction

- Added a focused research direction for eyeglasses, microphones, beard, local reflections, and mouth-region occluders.
- Interpretation:
  - These factors should not replace the broader multi-factor overfitting explanation.
  - They sit between identity/static appearance shortcut and local occlusion artifact.
  - They may act as subject identifiers, corrupt visible facial behavior regions, or create high-contrast ViT patches.
- Updated documentation:
  - `docs/RGB_OVERFITTING_AUDIT_PLAN.md`
  - `docs/SHORTCUT_AUDIT_DESIGN.md`
  - `docs/RESEARCH_NOTES.md`
  - `docs/TODO.md`
  - `docs/CURRENT_STATUS.md`
  - `docs/CODEX_CONTEXT.md`
- Recommended next step is case-study and spatial occlusion verification before adding global preprocessing rules.

### Systematic overfitting mechanism roadmap

- Added `docs/OVERFITTING_MECHANISM_ROADMAP.md` as a higher-level route for RGB overfitting mechanism research.
- The roadmap organizes current evidence into layers: data validity, prediction compression, input artifact/local occlusion, identity/static appearance, OpenFace geometry/quality, temporal/task context, and model optimization.
- It also adds a decision tree and stop rules to avoid scattered experiment accumulation.
- Updated references from `RGB_OVERFITTING_AUDIT_PLAN.md`, `TODO.md`, `CURRENT_STATUS.md`, and `CODEX_CONTEXT.md`.

### Temporal sampling ablation and overfit result review

- Reviewed temporal sampling prediction summary and training overfit summary.
- Main prediction result: `middle_crop` achieved the best overall temporal result with MAE about `8.80`, RMSE about `10.73`, Pearson about `0.42`, and CCC about `0.38`.
- `middle_crop` reduced severe underestimation but increased Freeform/Northwind task inconsistency.
- `uniform_256`, `uniform_512`, and `uniform_1024` were nearly identical, so insufficient uniform frame count is unlikely to be the main bottleneck.
- `first_crop` worsened prediction compression; `random_crop` increased prediction variance but was unstable.
- All temporal runs were still `overfit_after_best_val=True`; sampling replacement did not fix late training memorization.

### OpenFace boundary hard-transition smoothing plan

- Added a focused plan for boundary hard-transition smoothing.
- Priority variants: `edge_soften_only` and `border_blur_fill`.
- Goal: distinguish whether the RGB model is sensitive to black padding area itself or to the high-contrast transition between OpenFace black fill and face pixels.
- This remains an input artifact submechanism, not a replacement for identity, geometry, task context, or calibration analyses.

### Identity retrieval multi-run result review

- Reviewed identity retrieval outputs for `rgb`, `center_mask`, `center_mask_black_to_gray`, `border_black_feather`, and `middle_crop` on test and val splits.
- `rgb_test` showed strong identity retrieval: same-subject top-1 about `0.66`, top-5 about `0.85`, paired-task median rank `1`.
- `border_black_feather_test` had even stronger identity retrieval, with same-subject top-1 about `0.75` and top-5 about `0.90`, so artifact smoothing should not be interpreted as de-identification.
- `middle_crop_test` reduced same-subject top-1 to about `0.49`, but this coincided with worse task consistency; it likely reflects temporal/task-context confounding rather than a clean severity representation.
- No current variant simultaneously lowers identity retrieval, increases severity-neighbor agreement, improves BDI metrics, and preserves task consistency.
- Added follow-up task to build `scripts/summarize_identity_retrieval_runs.py` and a diagnostics module for multi-run identity retrieval summaries.

### RGB baseline severity calibration result review

- Reviewed `logs/severity_calibration` for the RGB baseline.
- Validation-only calibration fitted `pred_calibrated = 0.870402 * pred + 2.689282` on 100 validation records.
- Test original metrics: MAE `8.9145`, RMSE `10.9530`, Pearson `0.3526`, CCC `0.2925`, true std `11.48`, pred std `6.13`.
- Test calibrated metrics: MAE `8.8460`, RMSE `10.8278`, Pearson `0.3526`, CCC `0.2692`, pred std `5.33`.
- Severe underestimation remained almost unchanged: residual `-16.50 -> -16.10`; minimal overestimation worsened: `+6.89 -> +8.04`.
- Interpretation: RGB baseline has weak ranking signal but strong prediction range compression. Simple post-hoc linear calibration is not a solution; next work should run multi-run severity calibration summary and then test severity-aware training methods with identity/task-consistency checks.
### RGB input / identity / calibration joint summary review

- Reviewed `rgb_input_ablation_summary`, `identity_retrieval_summary`, and `severity_calibration_summary` together for `rgb`, `center_mask`, `border_black_feather`, `middle_crop`, and `center_mask_black_to_gray`.
- `center_mask_black_to_gray` had the best MAE (`7.726`) but the worst severe bias (`-17.46`), showing that overall MAE can be dominated by minimal/mild improvements.
- `center_mask` had the best CCC (`0.477`), largest prediction std (`8.132`), strong moderate-bias improvement, and near-baseline task consistency, making it the most stable input artifact mitigation evidence.
- `border_black_feather` reduced severe bias most (`-12.73`) but had the strongest identity retrieval (`same_top1=0.75`, `same_top5=0.90`), so boundary smoothing is not de-identification.
- `middle_crop` reduced identity retrieval (`same_top1=0.49`) but worsened task consistency and severity agreement, supporting temporal/task-context confounding.
- All linear calibration variants reduced CCC and compressed prediction variance, so post-hoc calibration remains diagnostic only.
### Revised feasible roadmap: identity suppression x boundary smoothing

- Reorganized the next research stage around a 2x2 mechanism ablation rather than more isolated RGB variants.
- The proposed factors are identity/static appearance suppression and OpenFace boundary hard-transition smoothing.
- Priority variants: `edge_soften_only`, `border_blur_fill`, `identity_texture_suppressed`, and `identity_texture_suppressed_edge_soften`.
- The goal is to determine whether identity shortcut and boundary artifact are independent mechanisms or coupled effects.
- Severity-aware training is deferred until this mechanism split is evaluated with prediction, identity retrieval, and severity calibration summaries.
### Identity suppression literature mapping

- Reviewed identity suppression, domain-adversarial learning, disentanglement, shortcut learning, and facial behavior representation methods.
- Mapped them to the current OpenFace aligned face setting.
- Current recommendation: prioritize input-level `identity_texture_suppressed` and boundary smoothing 2x2 ablation before subject-adversarial GRL or full disentanglement.
- Added explicit criteria for whether a method moves the model back toward depression-relevant behavior: identity retrieval should not rise, severity agreement and CCC should not drop, prediction std should not compress further, severe bias should improve, and task consistency should not degrade.

### Shortcut-Regularized MTL planning update

- Reframed the next project stage from additional input-filter expansion to a two-mechanism Shortcut-Regularized MTL route.
- Current formal mechanisms:
  - subject-level shortcut: handled by an identity-adversarial MTL branch with Gradient Reversal Layer;
  - severity-level label imbalance: handled by severity-balanced regression loss.
- Clarified that severity-balanced loss is intended to mitigate score-bin imbalance, not to directly maximize prediction variance.
- Moved dynamic facial-change features to Stage C as a later candidate direction, not part of the current experiment plan.
- Updated the planned experimental sequence:
  - Stage A closure: layer-wise identity probe and prediction error x identity similarity coupling;
  - Stage B intervention: E0 baseline, E1 severity-balanced regression, E2 identity-adversarial MTL, E3 combined;
  - Stage C deferred: feature delta / AU delta / landmark-pose-gaze delta / static-dynamic fusion.
- Updated `CURRENT_STATUS.md`, `RGB_OVERFITTING_AUDIT_PLAN.md`, `OVERFITTING_MECHANISM_ROADMAP.md`, `TODO.md`, and `CODEX_CONTEXT.md` accordingly.

### Documentation route clarity update

- Front-loaded the current authoritative route in `DOCS_GUIDE.md`, `CURRENT_STATUS.md`, `TODO.md`, `RGB_OVERFITTING_AUDIT_PLAN.md`, and `OVERFITTING_MECHANISM_ROADMAP.md`.
- Clarified that old RGB input ablations, boundary smoothing, temporal sampling, and identity texture suppression are historical evidence rather than the current main plan.
- Added a concrete immediate task block for Stage A and Stage B:
  - A1 layer-wise identity probe;
  - A2 prediction error x identity similarity coupling;
  - B1 severity-balanced regression;
  - B2 identity-adversarial MTL with GRL;
  - B3 baseline / severity / identity / combined four-run comparison.
- Explicitly deferred dynamic feature branches and new RGB input filters.

### Historical RPDF-Net mainline update (superseded)

- At that historical stage, promoted RPDF-Net (Risk-aware Progressive De-identification Factorization Network) to the future main research route.
- Repositioned Shortcut-Regularized MTL as a baseline and branch validation path rather than the final mainline.
- Updated the then-current staged plan:
  - Stage A: RPDF evidence closure with layer-wise identity probe, error-identity coupling, artifact weak-label audit, and severity imbalance summary;
  - Stage B: RPDF-lite single-level factorization `H0 -> z_dep,z_m,z_id,z_art,z_res`;
  - Stage C: two-level progressive factorization with controlled `z_m` transfer `H_k = Phi([z_dep^k, alpha_k * z_m^k])`;
  - Stage D: branch validation for `z_art`, controlled `z_m`, severity-balanced regression, multi-attacker privacy evaluation, and deferred dynamic features.
- Clarified that `z_m` must be controlled during layer transfer and final prediction, rather than freely propagated.
### Historical RPDF-Net route and literature consolidation (superseded)

- Consolidated the historical RPDF-Net route into four execution loops: evidence closure, RPDF-lite minimal model, progressive validation, and branch attribution.
- Added a literature-grounded rationale covering shortcut learning, small-sample ViT risk, DANN/GRL, disentanglement, imbalanced regression, OpenFace/LibreFace artifact variables, and facial behavior dynamics.
- Clarified that identity-adversarial MTL and severity-balanced regression are controlled baselines/branches, while RPDF-lite tests whether `z_dep/z_m/z_id/z_art/z_res` factorization is more appropriate than a single shared representation.
- Added fixed Stage B comparison groups and evaluation requirements: BDI metrics, severity bias, identity risk, artifact risk, task consistency, and train-val gap.

### 2026-07-06 Task-Nuisance route consolidation

- Revised the current research route from fine-grained RPDF-Net to **Shortcut-aware Task-Nuisance Disentangled Representation Learning**.
- New first-version representation split is `H0 -> z_dep, z_nuisance`, with optional `z_id` only when A1/A2 prove identity enters prediction and subject supervision is reliable.
- Demoted `z_art`, `z_m`, `z_ctx`, `z_quality`, and two-level progressive RPDF to historical design background or deferred branches rather than the current model mainline.
- Repositioned artifact/context/quality variables as shortcut probes, case-study anchors, and group-wise evaluation variables instead of first-version training branches.
- Updated route-facing docs and execution docs: `DOCS_GUIDE.md`, `CURRENT_STATUS.md`, `TODO.md`, `RGB_OVERFITTING_AUDIT_PLAN.md`, `OVERFITTING_MECHANISM_ROADMAP.md`, `CODEX_CONTEXT.md`, `RESEARCH_NOTES.md`, `SHORTCUT_AUDIT_DESIGN.md`, and `EXPERIMENT_SCRIPT_MANUAL.md`.

### 2026-07-14 Recent audit closure and AU-guided local-view route

- Closed four recent mechanism audits:
  - identity-adversarial gradients improved short-term validation utility but increased fresh external identity leakage and late overfit;
  - explicit L1/L2/elastic penalties produced nearly identical best validation RMSE and did not remove late overfit;
  - continuous severity-density weighting improved over plain MSE but remained weaker than the four-bin E2 baseline;
  - validation/test role swapping showed checkpoint-selection sensitivity, not a different training trajectory, and is exploratory because the original test split selected the swapped checkpoint.
- Did not authorize further representation-dimension or lambda sweeps. Dimension changes are not interpreted as feature separation without an independently verified semantic gradient or leakage reduction.
- Registered a new input-side intervention: one shared MTL-Lite model processes the global face and all valid AU/FACS semantic local views during training; validation, test, and inference remain global-only.
- Corrected the semantic granularity to four whole AU-related regions: brow, eye-cheek, nose-upper-lip, and mouth-jaw. Left/right landmarks remain internal tracking components for pose visibility and semantic-support composition, not separate views, predictions, or losses. The later 2026-07-17 contract supersedes the provisional blurred-mask input: masks are internal crop/coverage tools and the shared model receives rectangular local RGB crops.
- Added the required read-only tracking gate before training:
  - recover the coordinate mapping from OpenFace detection landmarks to the actual `112x112` aligned input;
  - compare static canonical, raw per-frame, and temporally stabilized dynamic masks;
  - audit valid-frame ratio, adjacent-mask IoU, centroid velocity, landmark jump rate, and pose/confidence-stratified failures;
  - inspect overlays from high-yaw, rapid-turn, low-confidence, and stable frontal cases.
- Frozen initial tracking gates: median adjacent-mask IoU `>=0.75`, valid-frame ratio `>=0.80`, landmark jump rate `<=0.05`, and no systematic high-yaw misalignment.
- Frozen falsification controls: global-only baseline, equal-area arbitrary grid local views, and AU-semantic local views under the same model and training budget. AU semantics may be claimed only if they outperform the grid control with global-only inference.
- Refined the implementation sequence after correcting AU granularity: frame join inventory and aligned-coordinate recovery precede tracking; four whole semantic masks are audited before model changes; a train-only 100-step AU-vs-grid gradient calibration and external identity probe precede full-40 training; seed 42 precedes seeds 43/44. The route explicitly forbids separate left/right views, region-specific heads, consistency-loss stacking, and multi-view inference.
- Implemented the AU-T0a frame-contract inventory as a read-only standard-library diagnostic. It matches normalized video ids across aligned frame directories and OpenFace CSV files, infers bounded frame offsets, detects parse failures/duplicates/missing frames/non-monotonic frame or timestamp sequences, and reproduces the configured temporal sampling indices. Added the CLI and focused synthetic tests. Local compile, CLI help, empty-root smoke, direct core/CLI synthetic tests, and `git diff --check` passed; local pytest collection is unavailable.

### 2026-07-15 Aligned-image identity gate and frozen OpenFace extraction protocol

- Completed the server-versus-Windows aligned-JPG comparison on the full dataset: `493,141/493,141` files are `EXACT_MATCH`, all images decode as `112x112 RGB JPEG`, both manifests have SHA-256 `07bc830c452a8faab1d58935d7f58d807313c218b3c4af8db053777c759daabb`, and the final gate is `EXACT_PASS`.
- Froze the Windows detector at `D:\Tools\OpenFace`, identified by the bundled README as OpenFace 2.2.0. Frozen hashes are `5995ae5cce749c4969ac4dd7e62d3f740cc9f702961f9573be7e14c4ca5b7f86` for `FeatureExtraction.exe` and `52f38548cffab1731f80e9e71f22a8b29373a2750eb6dc718069d56e82997543` for `model/main_ceclm_general.txt`.
- Added `scripts/run_openface_aligned_landmarks.ps1`. The extraction contract is one complete sorted `*_video_aligned` JPG sequence to one same-named CSV using only `-fdir`, `-out_dir`, `-2Dfp`, and `-mloc`; original RGB inputs and historical OpenFace features are never overwritten.
- The script records OpenFace binaries/models, script/git/PowerShell provenance, per-video logs, image/CSV row parity, continuous `frame=1..N`, required 68-point 2D columns, confidence summaries, and a minimum detection-success ratio of `0.995`. It supports a deterministic two-video debug and contract-aware `-Resume`.
- Tightened AU-T0b so `mapping_valid` also requires `success=1` from the actual mapping source. A failed OpenFace row can no longer pass merely because stale or invalid landmark coordinates remain in bounds; a focused regression test covers this case.
- Next gate: two-video extraction debug, then a clean full 300-video run, then AU-T0b with the new directory as `--aligned-openface-root`. Dynamic masks and training remain blocked until train-only overlays pass manual review.

### 2026-07-15 OpenFace two-video debug review and provenance correction

- Reviewed the copied Windows output for `203_1_Freeform` and `203_1_Northwind`. Both OpenFace processes reached 100% and closed successfully with no warning/error. CSV contracts passed at `930/930` and `990/990`; all 1,920 rows had `success=1`, continuous `frame=1..N`, the exact 68-point 2D schema, and no non-finite coordinates.
- Full-frame landmark in-bounds ratios were `0.99973` and `0.99893`; minimum per-frame ratios were `0.9853` and `0.9559`, above the frozen `0.80` threshold. Small excursions involved only the bottom/side jaw contour at the aligned-image boundary.
- Ran a temporary two-video T0b review against the local aligned JPGs and historical detection-space OpenFace CSVs: `2 REVIEW_REQUIRED / 0 FAIL / 0 BLOCKED`, all 192 selected frames valid, and all seven generated train overlays visually aligned across high-pose, rapid-change, low-confidence-relative, and frontal-control selections.
- Found one reproducibility defect: `D:\Project\stc` is a code copy without `.git`, so debug-v1 recorded empty git fields. The script hash exactly matched commit `1433a06`, preserving recoverability, but full extraction remains blocked until provenance is explicit.
- Added paired `SourceGitCommit/SourceGitBranch` parameters. Non-git copies now fail closed without them, detected checkouts reject explicit mismatches, and the actual Windows image root is frozen as `D:\Project\dataset\AVEC2014\face_images`. Next action is a fresh two-video debug-v2 with non-empty git provenance, then the full 300-video extraction.
