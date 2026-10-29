### PHerc0841 crop 1, trace (RTX 3090 run, 29 Oct)

Calibration: grey 56 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.208 | 0.329 | 40.32% | 1.00 |  | 0.850 | 0.707 |
| constant | blank | 0.224 | 0.047 | 71.34% | 1.77 | -0.013 | 0.504 | 0.252 |
| depth-shuffle | diagnostic | 0.196 | 0.086 | 17.35% | 0.43 | -0.015 | 0.551 | 0.262 |

Null threshold: grey 69 (99% quantile of the `constant` map, +1). 23.60% of the real map is above it; it keeps 65.2% of labelled ink at a 9.8% false-positive rate (calibration threshold: 80.1% at 27.1%).

- constant: 71.3% of structure-free input is above the calibration threshold (grey 56); that threshold cannot separate ink from model bias here.
