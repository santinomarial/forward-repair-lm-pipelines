# Table 2: Stratified Results — Query Corruption

| Stratum | Condition | EM | Recall@K | AllSupport@K |
| --- | --- | --- | --- | --- |
| Overall (excl. answer-not-in-support) | Baseline | 0.317 | 0.957 | 0.520 |
|  | Corrupted (query) | 0.113 ± 0.003 | 0.367 ± 0.025 | 0.058 ± 0.010 |
|  | Repaired (single) | 0.300 ± 0.006 | 0.937 ± 0.003 | 0.483 ± 0.015 |
|  | Repaired (iter.) | 0.333 | 0.967 | 0.533 |
|  |  |  |  |  |
| Single-hop-sufficient | Baseline | 0.229 | 0.960 | 0.484 |
|  | Corrupted (query) | 0.097 ± 0.005 | 0.399 ± 0.020 | 0.058 ± 0.013 |
|  | Repaired (single) | 0.236 ± 0.007 | 0.960 ± 0.004 | 0.462 ± 0.024 |
|  | Repaired (iter.) | 0.238 | 0.969 | 0.498 |
|  |  |  |  |  |
| Genuinely-multi-hop | Baseline | 0.571 | 0.948 | 0.623 |
|  | Corrupted (query) | 0.160 ± 0.007 | 0.273 ± 0.047 | 0.056 ± 0.020 |
|  | Repaired (single) | 0.485 ± 0.007 | 0.870 ± 0.013 | 0.545 ± 0.013 |
|  | Repaired (iter.) | 0.610 | 0.961 | 0.636 |
|  |  |  |  |  |
| Yes/No (comparison) | Baseline | 0.714 | 1.000 | 0.762 |
|  | Corrupted (query) | 0.000 | 0.016 ± 0.027 | 0.000 |
|  | Repaired (single) | 0.508 ± 0.027 | 0.905 | 0.540 ± 0.027 |
|  | Repaired (iter.) | 0.762 | 1.000 | 0.714 |
|  |  |  |  |  |
| Bridge (non-yes/no) | Baseline | 0.518 | 0.929 | 0.571 |
|  | Corrupted (query) | 0.220 ± 0.010 | 0.369 ± 0.057 | 0.077 ± 0.027 |
|  | Repaired (single) | 0.476 ± 0.021 | 0.857 ± 0.018 | 0.548 ± 0.010 |
|  | Repaired (iter.) | 0.554 | 0.946 | 0.607 |
|  |  |  |  |  |
