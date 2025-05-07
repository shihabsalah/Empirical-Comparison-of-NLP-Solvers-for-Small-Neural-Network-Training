## Performance Summary
| Optimizer | Loss | Accuracy | Loss / GD | Acc / GD | Time (s) | Mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| ADAM | 0.3898 | 88.5% | 0.17 | 5.26 | 17.42 | 416.1 |
| GRADIENT_DESCENT | 2.2597 | 16.8% | 1.00 | 1.00 | 17.33 | 415.7 |
| INTERIOR_POINT | 0.1806 | 95.1% | 0.08 | 5.65 | 308.02 | 437.8 |
| L_BFGS | 0.4815 | 97.5% | 0.21 | 5.80 | 18.93 | 426.6 |
| TRUST_REGION_NEWTON | 0.2164 | 97.8% | 0.10 | 5.81 | 38.01 | 469.4 |
