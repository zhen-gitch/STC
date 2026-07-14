# Continuous severity-density weighting audit

This audit tests a smoothed continuous alternative to the four-bin E2
severity-balanced MSE. The runner builds a train-only histogram for integer BDI
scores `0..MAX_SCORE`, smooths it with a Gaussian kernel (`sigma=2.0`), and
uses the bounded weight

```text
w(y) = clip((density_train(y) + epsilon)^(-alpha), 0.5, 4.0)
```

Weights are normalized by the empirical train-label distribution. Non-integer
labels use linear interpolation between adjacent score weights. Only the BDI
regression MSE is weighted; CCC and auxiliary losses remain unchanged.

The fixed first matrix contains two density exponents:

| run | alpha/power | sigma |
| --- | ---: | ---: |
| continuous_alpha025 | 0.25 | 2.0 |
| continuous_alpha05 | 0.50 | 2.0 |

Run the full 40-epoch validation-only matrix:

```bash
bash scripts/continuous_severity_weighting/run_matrix.sh
```

Smoke test:

```bash
DEBUG=1 SKIP_DIAG=1 bash scripts/continuous_severity_weighting/run_matrix.sh
```

Compare the generated overfit summary with the existing E2 results under
`logs/stage_b/aggregate/`. Do not use validation/test labels to alter the
density or add another exponent after inspecting the results.
