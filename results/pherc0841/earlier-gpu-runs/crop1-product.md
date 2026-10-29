### PHerc0841 crop 1, product (RTX 3090 run, 29 Oct)

Calibration: grey 21 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.063 | 0.314 | 37.43% | 1.00 |  | 0.841 | 0.711 |
| constant | blank | 0.035 | 0.008 | 0.00% | 0.00 | -0.003 | 0.502 | 0.250 |
| depth-shuffle | diagnostic | 0.031 | 0.027 | 1.44% | 0.04 | 0.115 | 0.534 | 0.263 |

Null threshold: grey 12 (99% quantile of the `constant` map, +1). 75.88% of the real map is above it; it keeps 92.2% of labelled ink at a 70.5% false-positive rate (calibration threshold: 80.0% at 23.3%).
