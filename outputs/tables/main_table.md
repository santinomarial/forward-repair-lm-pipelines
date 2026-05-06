# Table 1: Main Results

Error bars (± std) shown for conditions with 3 seeds (query corrupted, repaired single-shot).

Iterative and answer corruption conditions use seed 0 only.

| Condition | Exact Match | Contains Answer | Recall@K | AllSupport@K | Recovery Rate |
| --- | --- | --- | --- | --- | --- |
| Baseline (query) | 0.317 | 0.477 | 0.957 | 0.520 | — |
| Corrupted (query) | 0.113 ± 0.003 | 0.201 ± 0.012 | 0.367 ± 0.025 | 0.058 ± 0.010 | — |
| Repaired single-shot (query) | 0.300 ± 0.006 | 0.464 ± 0.010 | 0.937 ± 0.003 | 0.483 ± 0.015 | 0.246 ± 0.004 |
| Repaired iterative (query) | 0.333 | 0.477 | 0.967 | 0.533 | 0.273 |
|  |  |  |  |  |  |
| Corrupted (answer) | 0.077 | 0.650 | 0.957 | 0.520 | — |
| Repaired (answer) | 0.070 | 0.467 | 0.957 | 0.520 | 0.022 |
