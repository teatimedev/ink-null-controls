### PHerc0841 crop 1, A-hn seed 0

Calibration: grey 76 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.282 | 0.624 | 45.67% | 1.00 |  | 0.815 | 0.650 |
| constant | blank | 0.169 | 0.016 | 0.00% | 0.00 | 0.005 | 0.504 | 0.251 |
| phase-scramble | blank | 0.298 | 0.282 | 51.70% | 1.13 | -0.116 | 0.374 | 0.193 |
| pixel-shuffle | blank | 0.110 | 0.012 | 0.00% | 0.00 | 0.018 | 0.488 | 0.244 |
| block-shuffle | diagnostic | 0.282 | 0.537 | 46.21% | 1.01 | 0.636 | 0.684 | 0.493 |
| depth-shuffle | diagnostic | 0.157 | 0.055 | 0.23% | 0.01 | -0.030 | 0.450 | 0.223 |

Null threshold: grey 159 (99% quantile of the `phase-scramble` map, +1). 14.44% of the real map is above it; it keeps 42.6% of labelled ink at a 5.1% false-positive rate (calibration threshold: 80.6% at 34.1%).

- phase-scramble: 51.7% of structure-free input is above the calibration threshold (grey 76); that threshold cannot separate ink from model bias here.
- phase-scramble: output varies by 0.282 (p95-p05) on structure-free input; look for tile seams or invented structure.
- block-shuffle: as much or more area above threshold than the real input (1.01x). Area or mean confidence cannot rank regions here.
- At the null threshold (grey 159) only 43% of labelled ink survives.
