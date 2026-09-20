# Outcome routing and generalization results

Frozen outcome-trained policy. One provider seed; 60 held-out HotpotQA questions.

Costs and latency below cover selected incremental repair actions only.

## Seen

120 cases; 111 initially wrong; 9 initially correct.

| Policy | EM | Token F1 | Recovery | Damage | Calls/case | Tokens/case | Cost/case | Mean latency |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| accept | 7.5% | 20.0% | 0.0% | 0.0% | 0.00 | 0 | $0.000000 | 0.000s |
| rewrite_query | 35.0% | 51.5% | 32.4% | 33.3% | 2.00 | 1143 | $0.000190 | 1.889s |
| expand_context | 25.0% | 39.3% | 20.7% | 22.2% | 1.00 | 1685 | $0.000257 | 1.235s |
| regenerate | 24.2% | 35.0% | 18.0% | 0.0% | 1.00 | 911 | $0.000141 | 0.846s |
| learned | 36.7% | 52.5% | 33.3% | 22.2% | 1.32 | 1229 | $0.000196 | 1.030s |

Paired EM differences (learned minus comparator), percentage points:

- learned_minus_accept: +29.2 [18.3, 40.0]
- learned_minus_rewrite_query: +1.7 [-1.7, 5.0]
- learned_minus_expand_context: +11.7 [5.8, 18.3]
- learned_minus_regenerate: +12.5 [5.8, 19.2]

Learned-policy rates with 95% intervals:

| Metric | Estimate | 95% CI | Eligible cases |
|:--|--:|:--|--:|
| exact_match | 36.7% | [25.8%, 48.3%] | 120 |
| token_f1 | 52.5% | [42.7%, 62.5%] | 120 |
| recovery | 33.3% | [22.4%, 45.0%] | 111 |
| damage | 22.2% | [0.0%, 54.5%] | 9 |

## Unseen

120 cases; 93 initially wrong; 27 initially correct.

| Policy | EM | Token F1 | Recovery | Damage | Calls/case | Tokens/case | Cost/case | Mean latency |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| accept | 22.5% | 29.9% | 0.0% | 0.0% | 0.00 | 0 | $0.000000 | 0.000s |
| rewrite_query | 35.0% | 52.1% | 21.5% | 18.5% | 2.00 | 1147 | $0.000190 | 1.697s |
| expand_context | 36.7% | 52.5% | 21.5% | 11.1% | 1.00 | 1611 | $0.000224 | 2.647s |
| regenerate | 22.5% | 30.2% | 0.0% | 0.0% | 1.00 | 760 | $0.000118 | 1.316s |
| learned | 37.5% | 52.6% | 20.4% | 3.7% | 1.47 | 937 | $0.000152 | 1.386s |

Paired EM differences (learned minus comparator), percentage points:

- learned_minus_accept: +15.0 [8.3, 22.5]
- learned_minus_rewrite_query: +2.5 [-0.8, 6.7]
- learned_minus_expand_context: +0.8 [-5.0, 7.5]
- learned_minus_regenerate: +15.0 [8.3, 22.5]

Learned-policy rates with 95% intervals:

| Metric | Estimate | 95% CI | Eligible cases |
|:--|--:|:--|--:|
| exact_match | 37.5% | [25.8%, 50.0%] | 120 |
| token_f1 | 52.6% | [42.2%, 63.1%] | 120 |
| recovery | 20.4% | [11.8%, 31.0%] | 93 |
| damage | 3.7% | [0.0%, 13.0%] | 27 |

## Natural

60 cases; 39 initially wrong; 21 initially correct.

| Policy | EM | Token F1 | Recovery | Damage | Calls/case | Tokens/case | Cost/case | Mean latency |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| accept | 35.0% | 50.8% | 0.0% | 0.0% | 0.00 | 0 | $0.000000 | 0.000s |
| rewrite_query | 35.0% | 53.4% | 2.6% | 4.8% | 2.00 | 1152 | $0.000189 | 1.705s |
| expand_context | 36.7% | 54.4% | 5.1% | 4.8% | 1.00 | 1654 | $0.000199 | 2.765s |
| regenerate | 36.7% | 52.8% | 2.6% | 0.0% | 1.00 | 880 | $0.000133 | 0.705s |
| learned | 35.0% | 54.4% | 2.6% | 4.8% | 1.12 | 878 | $0.000131 | 1.165s |

Paired EM differences (learned minus comparator), percentage points:

- learned_minus_accept: +0.0 [-5.0, 5.0]
- learned_minus_rewrite_query: +0.0 [-5.0, 5.0]
- learned_minus_expand_context: -1.7 [-5.0, 0.0]
- learned_minus_regenerate: -1.7 [-6.7, 3.3]

Learned-policy rates with 95% intervals:

| Metric | Estimate | 95% CI | Eligible cases |
|:--|--:|:--|--:|
| exact_match | 35.0% | [23.3%, 46.7%] | 60 |
| token_f1 | 54.4% | [44.1%, 65.0%] | 60 |
| recovery | 2.6% | [0.0%, 8.3%] | 39 |
| damage | 4.8% | [0.0%, 15.8%] | 21 |

## Interpretation limits

Intervals are paired, question-clustered 95% percentile intervals (10,000 resamples). They are pointwise, not multiplicity-adjusted. Natural cases are unmodified benchmark outputs, not production data. All unseen variants come from the same HotpotQA corpus. Exact match and token F1 are lexical metrics; neither establishes evidence entailment. Latency reflects three-worker API collection, includes local rate-limit waits, and excludes router CPU overhead. The JSON separately records throttle time and measured wall time minus local throttle waits; these are not deployment benchmarks.

The JSON report includes per-family results, natural-error-only results, full intervals, action distributions, p95 latency, stage timings, and the difference between counterfactual collection cost and selected-policy cost.
