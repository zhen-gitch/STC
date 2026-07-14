# P0/P1/P2 Photometric Normalization

This matrix tests video-level luminance normalization without changing color
or the current C-REF/E2 training policy:

| Run | Mode | Intervention |
|---|---|---|
| P0 | `none` | Original RGB control |
| P1 | `luma_center` | Bounded video-level luminance median alignment |
| P2 | `luma_center_contrast` | P1 plus bounded q10-q90 contrast alignment |

The transform runs on raw RGB frames before existing input variants, resize,
model normalization, and padding.  It estimates one mapping from a deterministic
subset of each video's frames/pixels and applies that mapping to every frame.
Near-black OpenFace padding that reaches a border along a row or column is
excluded from the statistics and preserved in the output. P1/P2 add the same
luminance delta to all three RGB channels, so they do not intentionally
normalize color or skin tone. At saturated pixels, the shared delta is reduced
to the common in-gamut range instead of clipping channels independently; this
preserves RGB channel differences but can make the requested luminance target
only partially reachable.

`TARGET_MEDIAN=0.50` and `TARGET_SPAN=0.50` are preregistered canonical targets,
not statistics fitted from train, validation, or test.  This avoids split
leakage in the first bounded ablation.  P3 color constancy is intentionally not
part of this matrix.  The separate val/test role-swap audit lives under
`configs/split_sensitivity/` and does not add a fourth photometric condition.

All runs are validation-only (`RUN_TEST_AFTER_FIT=False`).  Keep the split,
backbone weights, seed, precision, and override order identical.  The external
DeiT weights must be configured in the machine-local `local_paths.yaml`.

```bash
python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p0_rgb.yaml

python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p1_luma_center.yaml

python scripts/train_mtl_lite.py \
  --override configs/photometric_normalization/common.yaml \
  --override configs/photometric_normalization/p2_luma_center_contrast.yaml
```

For a short data/model smoke, append:

```text
--override configs/photometric_normalization/debug_smoke.yaml
```

Do not interpret lower validation error alone as shortcut removal.  Compare
train-val gap, severity bias, task consistency, identity retrieval, and error
gaps grouped by the original per-video luminance statistics before opening the
test split.

The padding mask is deliberately conservative and is not a general connected-
component segmentation; bent or diagonal near-black paths can remain in the
statistics.  The transform falls back to the original video when fewer than
5% of sampled pixels are valid.  The current fixed model normalization
(`mean=std=0.5`) is also held identical across P0/P1/P2. Because the external
DeiT weights may expect ImageNet mean/std, model-specific normalization must be
tested later as a separate factorial control rather than changed in this
matrix.
