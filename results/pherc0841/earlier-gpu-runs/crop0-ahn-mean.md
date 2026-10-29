### PHerc0841 crop 0, ahn-mean (RTX 3090 run, 29 Oct)

Calibration: grey 125 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.373 | 0.651 | 35.57% | 1.00 |  | 0.896 | 0.835 |
| constant | blank | 0.165 | 0.004 | 0.00% | 0.00 | 0.003 | 0.502 | 0.294 |
| depth-shuffle | diagnostic | 0.165 | 0.055 | 0.00% | 0.00 | 0.032 | 0.465 | 0.272 |

Null threshold: grey 44 (99% quantile of the `constant` map, +1). 99.90% of the real map is above it; it keeps 100.0% of labelled ink at a 99.9% false-positive rate (calibration threshold: 80.0% at 17.1%).
