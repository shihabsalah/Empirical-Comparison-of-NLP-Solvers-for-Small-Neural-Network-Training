## Performance Summary
| Optimizer | Loss | Accuracy | Loss / GD | Acc / GD | Time (s) | Mem (MB) |
|---|---:|---:|---:|---:|---:|---:|
| ADAM | 0.3898 | 88.5% | 0.17 | 5.26 | 17.67 | 416.0 |
| GRADIENT_DESCENT | 2.2597 | 16.8% | 1.00 | 1.00 | 17.17 | 415.7 |
| INTERIOR_POINT | 0.1806 | 95.1% | 0.08 | 5.65 | 299.52 | 437.8 |
| L_BFGS | 0.2816 | 97.5% | 0.12 | 5.80 | 10.00 | 426.6 |
| TRUST_REGION_NEWTON | 0.2164 | 97.8% | 0.10 | 5.81 | 25.39 | 468.7 |
