### PHerc0841 crop 0, A-hn seed 0

Calibration: grey 118 keeps 80% of labelled ink on the real map.

| map | kind | median | p95−p05 | above calibration | × real | r vs real | AUC | AP |
|---|---|---|---|---|---|---|---|---|
| real | real | 0.333 | 0.663 | 36.56% | 1.00 |  | 0.899 | 0.828 |
| constant | blank | 0.169 | 0.016 | 0.00% | 0.00 | 0.009 | 0.506 | 0.296 |
| phase-scramble | blank | 0.286 | 0.267 | 5.49% | 0.15 | -0.141 | 0.414 | 0.245 |
| pixel-shuffle | blank | 0.114 | 0.016 | 0.00% | 0.00 | 0.016 | 0.507 | 0.297 |
| block-shuffle | diagnostic | 0.349 | 0.647 | 34.13% | 0.93 | 0.712 | 0.825 | 0.705 |
| depth-shuffle | diagnostic | 0.153 | 0.039 | 0.00% | 0.00 | 0.031 | 0.458 | 0.272 |

Null threshold: grey 151 (99% quantile of the `phase-scramble` map, +1). 29.17% of the real map is above it; it keeps 73.2% of labelled ink at a 10.9% false-positive rate (calibration threshold: 80.2% at 18.4%).

- phase-scramble: 5.5% of structure-free input is above the calibration threshold (grey 118); that threshold cannot separate ink from model bias here.
- phase-scramble: output varies by 0.267 (p95-p05) on structure-free input; look for tile seams or invented structure.
