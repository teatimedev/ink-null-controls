### PHerc0841 crop 0, product (RTX 3090 run, 29 Oct)

Calibration: grey 49 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.098 | 0.592 | 31.19% | 1.00 |  | 0.919 | 0.845 |
| constant | blank | 0.035 | 0.008 | 0.00% | 0.00 | -0.004 | 0.496 | 0.292 |
| depth-shuffle | diagnostic | 0.031 | 0.016 | 0.00% | 0.00 | 0.159 | 0.495 | 0.289 |

Null threshold: grey 12 (99% quantile of the `constant` map, +1). 86.33% of the real map is above it; it keeps 99.9% of labelled ink at a 80.7% false-positive rate (calibration threshold: 80.1% at 10.9%).
