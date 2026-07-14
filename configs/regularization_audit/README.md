# Explicit L1/L2 regularization audit

This fixed four-run audit measures whether explicit parameter penalties reduce
the full-40-epoch overfit already observed with the existing AdamW
`WEIGHT_DECAY: 5e-4`. All runs use the same seed, split, backbone and optimizer;
only the explicit loss coefficients differ:

| run | explicit L1 | explicit L2 |
| --- | ---: | ---: |
| reference | 0 | 0 |
| l1 | 0.01 | 0 |
| l2 | 0 | 0.1 |
| elastic | 0.01 | 0.1 |

The penalties are train-only, use the mean over trainable parameters with at
least two dimensions, and exclude bias/normalization vectors. Validation and
test monitor the unregularized BDI losses and metrics. Do not tune these
coefficients after inspecting validation results; the purpose is a falsifiable
fixed comparison.

Run on the server with:

```bash
bash scripts/regularization_audit/run_matrix.sh
```

For a smoke test, append `DEBUG=1`; to summarize existing runs without
training, use `SKIP_TRAIN=1`.
