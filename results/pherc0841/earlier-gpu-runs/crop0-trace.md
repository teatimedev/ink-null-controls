### PHerc0841 crop 0, trace (RTX 3090 run, 29 Oct)

Calibration: grey 96 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.278 | 0.573 | 34.16% | 1.00 |  | 0.910 | 0.811 |
| constant | blank | 0.224 | 0.047 | 0.00% | 0.00 | -0.009 | 0.494 | 0.290 |
| depth-shuffle | diagnostic | 0.196 | 0.082 | 0.00% | 0.00 | 0.157 | 0.512 | 0.303 |

Null threshold: grey 69 (99% quantile of the `constant` map, +1). 51.95% of the real map is above it; it keeps 94.7% of labelled ink at a 34.2% false-positive rate (calibration threshold: 80.2% at 15.0%).
