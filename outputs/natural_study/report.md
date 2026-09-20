# Natural-error repair: final held-out evaluation

300 fresh held-out HotpotQA questions; BM25; gpt-4o-mini; no injected corruption.

| Policy | EM | Recovery | Damage | Calls added | Cost added/question | Wall time added |
|:--|--:|--:|--:|--:|--:|--:|
| accept | 29.3% | 0.0% | 0.0% | 0.00 | $0.000000 | 0.00s |
| rewrite_query | 29.7% | 5.2% | 11.4% | 2.00 | $0.000182 | 2.39s |
| expand_context | 33.7% | 9.0% | 6.8% | 1.00 | $0.000229 | 0.99s |
| regenerate | 29.7% | 0.5% | 0.0% | 1.00 | $0.000129 | 0.98s |
| rewrite_query_10 | 33.0% | 8.5% | 8.0% | 2.00 | $0.000282 | 2.44s |
| preserve_evidence | 33.0% | 7.5% | 5.7% | 2.00 | $0.000279 | 2.96s |
| synthetic_original | 32.3% | 5.2% | 2.3% | 1.08 | $0.000118 | 1.21s |
| natural_original | 30.0% | 0.9% | 0.0% | 0.13 | $0.000023 | 0.10s |
| natural_augmented | 33.7% | 6.1% | 0.0% | 0.62 | $0.000092 | 0.49s |
| natural_damage_augmented | 34.3% | 7.5% | 1.1% | 0.91 | $0.000137 | 1.02s |

## Paired differences

The sole primary contrast is natural_damage_augmented minus accept; all others are exploratory.

| Contrast | EM difference | 95% CI | A-only / B-only correct |
|:--|--:|:--|--:|
| natural_damage_augmented_minus_accept | +5.0 pp | [2.3, 7.7] | 16 / 1 |
| preserve_evidence_minus_rewrite_query_10 | +0.0 pp | [-2.7, 2.7] | 8 / 8 |
| preserve_evidence_minus_expand_context | -0.7 pp | [-3.3, 2.0] | 7 / 9 |
| natural_original_minus_synthetic_original | -2.3 pp | [-5.0, 0.0] | 4 / 11 |
| natural_augmented_minus_natural_original | +3.7 pp | [1.3, 6.0] | 12 / 1 |
| natural_damage_augmented_minus_natural_augmented | +0.7 pp | [-0.7, 2.0] | 3 / 1 |
| natural_damage_augmented_minus_rewrite_query | +4.7 pp | [1.3, 8.3] | 22 / 8 |
| natural_damage_augmented_minus_expand_context | +0.7 pp | [-2.0, 3.3] | 10 / 8 |

## Recovery and damage uncertainty

Primary-policy rates use paired bootstrap intervals in JSON, supplemented here with Wilson binomial intervals. Zero observed events do not imply zero population risk.

- Recovery: 7.5%; Wilson 95% CI [4.7%, 11.9%]; denominator 212.
- Damage: 1.1%; Wilson 95% CI [0.2%, 6.2%]; denominator 88.

## Limits

One model seed, one benchmark, 300 held-out questions. New pooled corpus and bounded passages differ from the historical study, so raw EM is not a historical regression comparison. Equal token/document ceilings do not imply equal realized context length. Exact match uses the project's leading yes/no convention. Bootstrap intervals are pointwise, not multiplicity-adjusted; secondary contrasts are exploratory. Natural errors mean failures of unmodified benchmark outputs, not production incidents. No human semantic review is claimed.

Costs are selected-policy estimates from offline observed actions, not an online deployment. Wall times include provider/local pacing and exclude router CPU; unthrottled and baseline-inclusive totals are in JSON. Shared query generation is charged once per selected rewrite policy but only once in actual collection cost.

All EM-change cases for the primary policy are exported to review_queue.jsonl. Human semantic review remains pending; lexical improvement is not proof of factuality.
