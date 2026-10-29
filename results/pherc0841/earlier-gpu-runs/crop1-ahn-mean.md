### PHerc0841 crop 1, ahn-mean (RTX 3090 run, 29 Oct)

Calibration: grey 83 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.310 | 0.612 | 46.37% | 1.00 |  | 0.807 | 0.642 |
| constant | blank | 0.165 | 0.004 | 0.00% | 0.00 | -0.001 | 0.501 | 0.249 |
| depth-shuffle | diagnostic | 0.169 | 0.102 | 1.68% | 0.04 | 0.099 | 0.492 | 0.263 |

Null threshold: grey 44 (99% quantile of the `constant` map, +1). 99.79% of the real map is above it; it keeps 100.0% of labelled ink at a 99.7% false-positive rate (calibration threshold: 80.0% at 35.2%).
