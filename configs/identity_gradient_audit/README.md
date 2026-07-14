# Identity Gradient Audit

This experiment tests whether a continuously trained identity attacker gives
the regression-only representation a useful adversarial gradient, whether the
BDI and reversed-identity gradients conflict, and whether the adversarial run
changes the full-40 overfit trajectory.

The paired runs share seed, split, backbone, optimizer, precision, maximum
epochs, checkpoint policy, and pure MSE regression. Test remains closed.

```text
iga_regression_only_reference:
  H0 -> BDI

iga_bdi_identity_adversarial:
  H0 -> BDI
  H0 -> GRL(lambda=0.05) -> train-subject classifier
```

The scalar loss remains `L_BDI + L_identity`; GRL alone reverses and scales the
identity gradient seen by the representation. Do not subtract identity CE from
the global loss, because that would also train the attacker in the wrong
direction.

Logged gradient fields include BDI norm, reversed-identity norm, their ratio,
cosine, conflict indicator, cancellation fraction, and train attacker
accuracy. Negative cosine indicates conflict at the shared representation.

Run the complete paired workflow with:

```bash
bash scripts/identity_gradient_audit/run_matrix.sh
```

Before the full run, verify the candidate path and gradient columns with:

```bash
python scripts/train_mtl_lite.py \
  --override configs/stage_b/base_regression_only.yaml \
  --override configs/identity_gradient_audit/common_full40.yaml \
  --override configs/identity_gradient_audit/bdi_identity_adversarial.yaml \
  --override configs/identity_gradient_audit/debug_smoke.yaml
```

The debug `metrics.csv` must contain finite `train_grad_*` epoch/step fields
before the full paired matrix is authorized.

Environment switches:

```text
SKIP_TRAIN=1  reuse completed runs
SKIP_DIAG=1   skip train/val representation and group diagnostics
```

The workflow writes four read-only analysis families under the experiment
group's `analysis/` directory:

```text
gradient_conflict/
training_overfit/
representation_leakage/
group_robustness/
```

`gradient_conflict` reports representation-level gradient interaction.
`training_overfit` compares best-validation and last-epoch behavior across the
paired runs. Representation and group audits fit thresholds/probes on train and
evaluate validation; test is never loaded.

This is a bounded gradient-mechanism experiment, not a reopened Stage C gate.
No lambda or dimension sweep is allowed before the paired result is reviewed.
