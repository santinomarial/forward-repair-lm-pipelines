# Outcome-based routing and held-out generalization

## Protocol fixed before collection

This study completes two experiments: learning which repair is useful and
testing that policy on held-out questions, unseen perturbations, and natural
benchmark errors. It is a bounded, single-provider-seed study, not a production
reliability claim or a cross-dataset evaluation.

## Completed results

All 480 development and 300 test cases are collected. The frozen policy improved
EM over no repair by 29.2 percentage points on seen failures (95% CI 18.3–40.0)
and 15.0 points on unseen perturbations (8.3–22.5). Gains over always rewriting
were small and inconclusive. On natural outputs it recovered one of 39 initial
EM errors but damaged one of 21 correct answers: no net EM improvement.

Read the [full comparison and confidence intervals](../outputs/reliability_study/report.md)
or [JSON summary](../outputs/reliability_study/summary.json). Completed cases used
3,240 calls and 2,918,437 tokens, with estimated cost $0.4548963. The persistent
ledger holds 3,250 reservations totaling $1.432373625; the ten extra reservations
cover failed or discarded work from the development interruption. Completed-case
cost excludes that work, while the reservation total conservatively includes it.
The original $3 ceiling was never raised.

The two unseen families differ: EM rises from 30.0% to 40.0% for query-term
substitution and from 15.0% to 35.0% for rank dropout. These are two mechanisms
within one benchmark, not a broad out-of-distribution guarantee. Exact match
and token F1 do not independently establish factuality.

## Study design

The existing 300 HotpotQA questions are split by question ID with seed 17:
180 training, 60 validation, 60 test. All variants of a question stay in its
split. The corpus is a shared retrieval knowledge base; no gold answer or
support annotation is an inference feature.

### Development data

Training and validation use two saved failure mechanisms: vague-query corruption
and ignoring the retrieved context during answer generation. Neither the old
stage-attribution router nor its labels are reused. Every state receives all
four actions so that their observed outcomes and costs can supervise the new
controller:

| Action | Intervention | Expected incremental calls |
|:--|:--|--:|
| Accept | Keep the initial answer | 0 |
| Rewrite query | Repair query, retrieve top 5, generate a fresh answer | 2 |
| Expand context | Same query, retrieve top 10, generate a fresh answer | 1 |
| Regenerate | Same evidence, fresh answer without the previous answer | 1 |

All repair arms use the existing concise, evidence-grounded answer prompt.
Action order is deterministically randomized per case. Settings: GPT-4o-mini,
temperature 0, provider seed 0, 300 output tokens, DSPy response cache disabled.
Each worker has its own LM history, so concurrent requests do not contaminate
per-action usage. Three workers run independent cases.

### Policy and model selection

The controller is a small multi-output ridge regression model. Targets are
observed `exact_match - 100 * incremental_cost_usd` for each action. Thus a
$0.0002 action incurs a 0.02 utility penalty. This coefficient is fixed before
collection, not tuned on test results. Exact match is an operational proxy for
success, not a validated semantic factuality label.

Features are eight existing lexical failure signals plus answer length, query
length, and document count. Family labels, question IDs, gold answers, support
labels, and candidate action outcomes are not available to inference. Feature
normalization is fitted only on training rows.

Validation selects ridge strength from `[0.1, 1, 10, 100]` and minimum predicted
gain over accepting from `[0, 0.05, 0.10]`. Selection maximizes mean observed
validation utility; ties prefer lower cost, then higher minimum gain. The model
is then serialized and hash-locked before any test action outcomes are collected.
Test outcomes cannot be used to refit the locked policy.

The completed development phase contains 360 training and 120 validation cases.
Validation selected ridge strength **100** and minimum predicted gain **0.10**.
The frozen [model](../outputs/reliability_study/model.json) includes all 12
validation candidates and hashes of the development data and protocol manifest.

### Held-out evaluation

Each of the 60 test questions contributes five states:

1. Saved vague-query corruption (seen mechanism).
2. Saved ignore-context corruption (seen mechanism).
3. **Query-term substitution:** replace the longest alphabetic query term with
   a deterministic sampled title from outside the initial retrieved set, then
   retrieve and answer. This is an unseen perturbation mechanism, not a claim
   that the substituted term is always a named entity.
4. **Retrieval-rank dropout:** remove the first two retrieved documents, then
   answer from the remaining three. Repairs can access the intact corpus.
5. **Natural:** retain the unmodified historical baseline output. Evaluate all
   cases, including initially correct answers, and separately report its
   naturally occurring exact-match failures. These are benchmark pipeline
   errors, not real production traffic.

Development has 480 cases; test has 300 cases. Report the learned policy against
no repair, always rewriting, always expanding context, and always regenerating.
The action table permits paired offline evaluation without extra LLM calls:
the learned policy selects from observed outcomes using only initial features.
This is full-information policy evaluation, not an online deployment trial.

## Metrics and statistical plan

Report exact match, normalized token F1, contains-answer, wrong-to-right recovery,
right-to-wrong damage, incremental calls, tokens, estimated cost, mean/p95
latency, and action frequencies. Recovery and damage denominators are the same
initially wrong/correct cases for every policy; empty denominators are null.
Token F1 is an additional lexical check for response-format effects, not an
entailment or semantic-correctness evaluator.

Use 10,000 paired cluster-bootstrap samples by question ID, keeping every variant
of a sampled question together. Report 95% pointwise intervals on rates and
paired differences. Primary comparisons are learned versus accept on seen
held-out cases, unseen mechanisms, and natural cases; other contrasts are
exploratory. There is no multiplicity correction. Do not select a policy after
reading test performance.

Cost comparisons count only selected incremental interventions. Historical
initial-pass costs are unavailable and excluded consistently; generating the
new perturbation states is recorded separately as study setup cost. Total
collection cost includes all counterfactual actions, not only the actions the
learned policy would choose. Latency is a three-worker API experiment and is
not a production serving benchmark.

After a provider token-per-minute rejection during development collection, a
shared rolling-minute limiter was added at 180,000 tokens/minute. Completed
cases were retained and the interrupted collection resumed without prompt or
selection changes. Measured wall/stage times include local throttle waits;
the final JSON also reports throttle time and wall time minus those waits.
Earlier development records predate this instrumentation. All held-out test
records use the same paced collector. Pacing affects timing, not the fixed
quality-evaluation protocol.

## Spending and recovery

A persistent ledger reserves uncached input cost plus the maximum output cost
before each request, with token padding and 25% headroom. Reservations are never
refunded, including failed requests. The combined development/test reservation
cap is **$3**, using the verified GPT-4o-mini list prices of $0.15/$0.60 per
million input/output tokens. This is an experiment-level guard, not an account
balance query or an account-wide spending limit.

Automatic transport retries are disabled. DSPy adapter fallbacks must reserve
again. Concurrent collectors using the same study directory are blocked.
Completed cases are checkpointed; resumes verify source, prompt, and manifest
hashes, retain prior reservations, and skip completed cases. A damaged JSONL
record causes a failure instead of being silently discarded.

## Commands

```bash
# Rebuild the published report offline; no paid calls.
python src/reliability_study.py report

# A new paid replication needs a fresh directory, not the published artifacts.
python src/reliability_study.py collect --directory outputs/reliability_replication --limit 2
python src/reliability_study.py collect --directory outputs/reliability_replication --resume

# Fit once, then lock the model before collecting test outcomes.
python src/reliability_study.py fit --directory outputs/reliability_replication
python src/reliability_study.py collect --directory outputs/reliability_replication --phase test

# Offline report: no API key or paid calls required.
python src/reliability_study.py report --directory outputs/reliability_replication
```

The default artifact directory is `outputs/reliability_study/`. Artifacts include
the protocol manifest and input hashes, persistent budget ledger, complete action
outcomes, frozen model and validation selection, and final metrics. Keep the
same directory and use `--resume` after an interruption. Use `--limit` for a
smoke run; fitting and final reporting reject incomplete planned datasets.
